import os
import spacy
import networkx as nx
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import json
import numpy as np
from pyvis.network import Network
import secrets
from difflib import SequenceMatcher
import requests
import xml.etree.ElementTree as ET
import wikipedia
import arxiv
import re
from time import sleep
from flask import send_file, render_template, abort
import os
import mimetypes
# NLP imports
from nlp.preprocessing import preprocess_text
from nlp.ner import extract_entities
from nlp.relation_extraction import extract_relations
from nlp.graph_builder import build_knowledge_graph, get_subgraph
from nlp.semantic_search import semantic_search, initialize_encoder

# ============================================
# 1. INITIALIZE FLASK APP FIRST (MOST IMPORTANT!)
# ============================================
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///knowledge_graph.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================
# 2. INITIALIZE DATABASE AND LOGIN MANAGER
# ============================================
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load NLP models
nlp = spacy.load("en_core_web_sm")
encoder = initialize_encoder()

# ============================================
# 3. DATABASE MODELS
# ============================================
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    datasets = db.relationship('Dataset', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'

class Dataset(db.Model):
    __tablename__ = 'datasets'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    domain = db.Column(db.String(50), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed = db.Column(db.Boolean, default=False)
    source_type = db.Column(db.String(50), default='upload')  # 'upload', 'wikipedia', 'arxiv'
    source_url = db.Column(db.String(500), nullable=True)  # Original source URL
    
    # Relationships
    entities = db.relationship('Entity', backref='dataset', lazy=True, cascade='all, delete-orphan')
    relations = db.relationship('Relation', backref='dataset', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Dataset {self.name}>'

class Entity(db.Model):
    __tablename__ = 'entities'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    confidence = db.Column(db.Float, default=1.0)
    merged_with = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Entity {self.name} ({self.type})>'

class Relation(db.Model):
    __tablename__ = 'relations'
    
    id = db.Column(db.Integer, primary_key=True)
    entity1_id = db.Column(db.Integer, db.ForeignKey('entities.id'), nullable=False)
    entity2_id = db.Column(db.Integer, db.ForeignKey('entities.id'), nullable=False)
    relation_type = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, default=1.0)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    approved = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    entity1 = db.relationship('Entity', foreign_keys=[entity1_id])
    entity2 = db.relationship('Entity', foreign_keys=[entity2_id])
    
    def __repr__(self):
        return f'<Relation {self.entity1.name if self.entity1 else None} - {self.relation_type} - {self.entity2.name if self.entity2 else None}>'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ============================================
# 4. ALL ROUTES GO HERE (AFTER app IS DEFINED)
# ============================================

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    datasets = Dataset.query.filter_by(user_id=current_user.id).all()
    
    # Calculate cross-domain stats
    cross_domain_datasets = 0
    cross_domain_relations = 0
    
    for dataset in datasets:
        if any(r.relation_type in ['same_as', 'related_to'] for r in dataset.relations):
            cross_domain_datasets += 1
    
    cross_domain_relations = Relation.query.filter(
        Relation.dataset_id.in_([d.id for d in datasets]),
        Relation.relation_type.in_(['same_as', 'related_to', 'works_for', 'employs', 'lives_in', 'located_in'])
    ).count()
    
    # Count external source datasets
    wikipedia_datasets = sum(1 for d in datasets if d.source_type == 'wikipedia')
    arxiv_datasets = sum(1 for d in datasets if d.source_type == 'arxiv')
    
    stats = {
        'total_datasets': len(datasets),
        'total_entities': Entity.query.join(Dataset).filter(Dataset.user_id == current_user.id).count(),
        'total_relations': Relation.query.join(Dataset).filter(Dataset.user_id == current_user.id).count(),
        'cross_domain_datasets': cross_domain_datasets,
        'cross_domain_relations': cross_domain_relations,
        'wikipedia_datasets': wikipedia_datasets,
        'arxiv_datasets': arxiv_datasets
    }
    
    return render_template('dashboard.html', datasets=datasets, stats=stats)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        domain = request.form.get('domain')
        
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and file.filename.endswith(('.txt', '.csv')):
            filename = secure_filename(f"{current_user.id}_{datetime.now().timestamp()}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Create dataset record
            dataset = Dataset(
                name=file.filename,
                domain=domain,
                filename=filename,
                user_id=current_user.id,
                source_type='upload'
            )
            db.session.add(dataset)
            db.session.commit()
            
            # Process the dataset
            process_dataset(dataset.id, filepath)
            
            flash('Dataset uploaded and processing started!')
            return redirect(url_for('dashboard'))
    
    return render_template('upload.html')
    

# Alternative more efficient version using SQLAlchemy queries:
@app.route('/datasets')
@login_required
def datasets():
    # Get current user's datasets
    datasets = Dataset.query.filter_by(user_id=current_user.id).all()
    
    # Efficiently calculate stats using SQLAlchemy queries
    total_entities = db.session.query(db.func.sum(db.func.length(Dataset.entities))).filter(
        Dataset.user_id == current_user.id
    ).scalar() or 0
    
    total_relations = db.session.query(db.func.sum(db.func.length(Dataset.relations))).filter(
        Dataset.user_id == current_user.id
    ).scalar() or 0
    
    # Get datasets with cross-domain relations
    cross_domain_datasets = Dataset.query.filter(
        Dataset.user_id == current_user.id,
        Dataset.relations.any(Relation.relation_type.in_(['same_as', 'related_to']))
    ).count()
    
    stats = {
        'total_entities': total_entities,
        'total_relations': total_relations,
        'cross_domain_datasets': cross_domain_datasets
    }
    
    return render_template('datasets.html', datasets=datasets, stats=stats)
# ============================================
# 5. WIKIPEDIA AND ARXIV ROUTES (FIXED)
# ============================================

# ── Scoring-based relevance check ──────────────────────────────────────────
# Instead of "any keyword matches → pass", we require a minimum SCORE from
# BOTH the healthcare bucket AND the software bucket. Single generic words
# like "health", "data", "model" are removed; only compound/specific phrases
# and strong single terms are kept.

# High-value healthcare phrases (each scores 2 points)
HEALTHCARE_PHRASES = [
    'electronic health record', 'electronic medical record', 'ehr', 'emr',
    'clinical decision support', 'clinical trial', 'patient outcome',
    'medical imaging', 'disease prediction', 'diagnostic accuracy',
    'telemedicine', 'telehealth', 'precision medicine', 'personalized medicine',
    'hospital readmission', 'sepsis', 'radiology', 'pathology', 'oncology',
    'cardiology', 'neurology', 'epidemiology', 'public health surveillance',
    'drug discovery', 'pharmacovigilance', 'adverse drug', 'patient triage',
    'medical diagnosis', 'clinical outcome', 'biomedical', 'bioinformatics',
    'genomics', 'medical record', 'health informatics', 'clinical informatics',
]

# High-value software/AI phrases (each scores 2 points)
SOFTWARE_PHRASES = [
    'machine learning', 'deep learning', 'neural network', 'convolutional neural',
    'random forest', 'support vector machine', 'natural language processing',
    'large language model', 'transformer model', 'federated learning',
    'computer vision', 'image segmentation', 'object detection',
    'predictive model', 'classification model', 'regression model',
    'decision tree', 'gradient boosting', 'xgboost', 'reinforcement learning',
    'knowledge graph', 'data pipeline', 'data warehouse', 'real-time analytics',
    'clinical nlp', 'medical nlp', 'medical image analysis',
    'artificial intelligence', 'explainable ai', 'xai',
]

# Weaker single-word signals (each scores 1 point) — need multiples to matter
HEALTHCARE_SINGLE = [
    'patient', 'clinical', 'diagnosis', 'treatment', 'physician',
    'hospital', 'disease', 'symptom', 'therapy', 'medication',
    'vaccine', 'surgical', 'chronic', 'acute', 'mortality',
]

SOFTWARE_SINGLE = [
    'algorithm', 'dataset', 'benchmark', 'inference', 'training',
    'classifier', 'accuracy', 'precision', 'recall', 'f1-score',
    'automation', 'pipeline', 'framework', 'deployment', 'model',
]

# Minimum scores required (moderate strictness)
MIN_HEALTHCARE_SCORE = 3   # e.g. one phrase (2) + one single word (1)
MIN_SOFTWARE_SCORE   = 3


def score_text(text: str, phrases: list, singles: list) -> int:
    """Return a relevance score for one bucket (healthcare or software)."""
    score = 0
    for phrase in phrases:
        if phrase in text:
            score += 2
    for word in singles:
        # Use word-boundary-like check: surrounded by non-alpha chars
        if f' {word} ' in text or text.startswith(word + ' ') or text.endswith(' ' + word):
            score += 1
    return score


def is_healthcare_software_related(text: str, title: str, categories=None) -> bool:
    """
    Returns True only when BOTH the healthcare score AND the software score
    meet their minimum thresholds.  Uses compound phrases as primary signal
    to avoid false positives from generic single words.
    """
    combined = (title + ' ' + text).lower()
    if categories:
        combined += ' ' + ' '.join(categories).lower()

    h_score = score_text(combined, HEALTHCARE_PHRASES, HEALTHCARE_SINGLE)
    s_score = score_text(combined, SOFTWARE_PHRASES,   SOFTWARE_SINGLE)

    return h_score >= MIN_HEALTHCARE_SCORE and s_score >= MIN_SOFTWARE_SCORE


# ── Wikipedia routes ────────────────────────────────────────────────────────

@app.route('/import/wikipedia', methods=['POST'])
@login_required
def import_wikipedia():
    """Import healthcare-software articles from Wikipedia (fixed filtering)."""
    search_query = request.form.get('search_query')

    if not search_query:
        flash('Please enter a search query')
        return redirect(url_for('dashboard'))

    try:
        search_results = wikipedia.search(search_query, results=15)  # wider net

        if not search_results:
            flash(f'No Wikipedia articles found for "{search_query}"')
            return redirect(url_for('dashboard'))

        imported_articles = []

        for article_title in search_results:
            try:
                page = wikipedia.page(article_title, auto_suggest=False)
                content = page.content

                if not is_healthcare_software_related(content, article_title, page.categories):
                    print(f"[SKIP] '{article_title}' — does not meet healthcare-software threshold")
                    sleep(0.3)
                    continue

                safe_title = re.sub(r'[^\w\s-]', '', article_title).strip().replace(' ', '_')
                filename = secure_filename(
                    f"{current_user.id}_{datetime.now().timestamp()}_{safe_title}.txt"
                )
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Title: {article_title}\n\n")
                    f.write(f"URL: {page.url}\n\n")
                    f.write(f"Categories: {', '.join(page.categories[:10])}\n\n")
                    f.write("=== HEALTHCARE-SOFTWARE CONTENT ===\n\n")
                    f.write(content)

                dataset = Dataset(
                    name=f"Wikipedia (Healthcare-Software): {article_title}",
                    domain='healthcare_software',
                    filename=filename,
                    user_id=current_user.id,
                    source_type='wikipedia',
                    source_url=page.url
                )
                db.session.add(dataset)
                db.session.commit()
                process_dataset(dataset.id, filepath)

                imported_articles.append(article_title)

                if len(imported_articles) >= 3:
                    break

                sleep(0.5)

            except Exception as e:
                print(f"Error checking/importing '{article_title}': {e}")
                continue

        if imported_articles:
            flash(
                f'Successfully imported {len(imported_articles)} healthcare-software articles: '
                f'{", ".join(imported_articles)}'
            )
        else:
            flash(
                f'No qualifying healthcare-software articles found for "{search_query}". '
                f'Try queries like "clinical NLP", "medical image segmentation", '
                f'"health informatics AI", or "EHR machine learning".'
            )

    except wikipedia.exceptions.DisambiguationError as e:
        flash(f'Disambiguation error. Did you mean: {", ".join(e.options[:5])}?')
    except wikipedia.exceptions.PageError:
        flash(f'Wikipedia page not found for "{search_query}"')
    except Exception as e:
        flash(f'Error importing Wikipedia article: {str(e)}')

    return redirect(url_for('dashboard'))


@app.route('/import/wikipedia/advanced', methods=['POST'])
@login_required
def import_wikipedia_advanced():
    """Advanced Wikipedia import with improved healthcare-software filtering."""
    query  = request.form.get('query')
    limit  = int(request.form.get('limit', 5))

    if not query:
        flash('Please enter a search query')
        return redirect(url_for('dashboard'))

    try:
        search_results = wikipedia.search(query, results=limit * 3)  # wider net

        if not search_results:
            flash(f'No Wikipedia articles found for "{query}"')
            return redirect(url_for('dashboard'))

        imported_count = 0

        for article_title in search_results:
            try:
                page    = wikipedia.page(article_title, auto_suggest=False)
                content = page.content

                if not is_healthcare_software_related(content, article_title, page.categories):
                    print(f"[SKIP] '{article_title}' — does not meet threshold")
                    sleep(0.3)
                    continue

                safe_title = re.sub(r'[^\w\s-]', '', article_title).strip().replace(' ', '_')
                filename   = secure_filename(
                    f"{current_user.id}_{datetime.now().timestamp()}_{safe_title}.txt"
                )
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Title: {article_title}\n\n")
                    f.write(f"URL: {page.url}\n\n")
                    f.write(f"Categories: {', '.join(page.categories[:10])}\n\n")
                    f.write("=== HEALTHCARE-SOFTWARE CONTENT ===\n\n")
                    f.write(content)

                dataset = Dataset(
                    name=f"Wikipedia (Healthcare-Software): {article_title}",
                    domain='healthcare_software',
                    filename=filename,
                    user_id=current_user.id,
                    source_type='wikipedia',
                    source_url=page.url
                )
                db.session.add(dataset)
                db.session.commit()
                process_dataset(dataset.id, filepath)

                imported_count += 1
                if imported_count >= limit:
                    break

                sleep(0.5)

            except Exception as e:
                print(f"Error importing '{article_title}': {e}")
                continue

        if imported_count > 0:
            flash(f'Successfully imported {imported_count} healthcare-software Wikipedia articles')
        else:
            flash(
                f'No qualifying articles found for "{query}". '
                f'Try "health informatics", "clinical AI", or "medical NLP".'
            )

    except Exception as e:
        flash(f'Error in advanced Wikipedia import: {str(e)}')

    return redirect(url_for('dashboard'))


# ── arXiv routes ────────────────────────────────────────────────────────────

# Tightly scoped arXiv categories for healthcare + AI/ML
_ARXIV_HEALTH_AI_CATS = [
    'cs.LG',          # Machine Learning
    'cs.AI',          # Artificial Intelligence
    'cs.CY',          # Computers and Society (health informatics)
    'cs.HC',          # Human-Computer Interaction
    'eess.IV',        # Image and Video Processing (medical imaging)
    'q-bio.QM',       # Quantitative Methods / bioinformatics
    'stat.AP',        # Statistics Applications (biostatistics)
    'physics.med-ph', # Medical Physics
]

# arXiv query: must mention both a health term AND an AI/ML term
# Using AND logic so both must appear — much stricter than the original OR
_ARXIV_HEALTH_FILTER  = '(healthcare OR "electronic health record" OR "clinical trial" OR "medical imaging" OR "patient outcome" OR "disease prediction" OR "biomedical" OR "health informatics")'
_ARXIV_SOFTWARE_FILTER = '(\"machine learning\" OR \"deep learning\" OR \"neural network\" OR \"natural language processing\" OR \"computer vision\" OR \"federated learning\" OR \"large language model\")'


def _build_arxiv_query(user_query: str, category: str = 'all') -> str:
    """Build a tight arXiv query that requires both health AND AI/ML terms."""
    cat_filter = (
        f'cat:{category}'
        if category and category != 'all'
        else '(' + ' OR '.join(f'cat:{c}' for c in _ARXIV_HEALTH_AI_CATS) + ')'
    )
    return (
        f'({cat_filter}) AND ({user_query}) '
        f'AND {_ARXIV_HEALTH_FILTER} AND {_ARXIV_SOFTWARE_FILTER}'
    )


def _arxiv_paper_passes(result) -> bool:
    """Secondary check: run scoring filter on title + abstract."""
    combined = (result.title + ' ' + result.summary).lower()
    h = score_text(combined, HEALTHCARE_PHRASES, HEALTHCARE_SINGLE)
    s = score_text(combined, SOFTWARE_PHRASES,   SOFTWARE_SINGLE)
    return h >= MIN_HEALTHCARE_SCORE and s >= MIN_SOFTWARE_SCORE


def _save_arxiv_paper(result, label: str = 'Healthcare-Software') -> None:
    """Persist one arXiv result to disk + DB and trigger processing."""
    content  = f"Title: {result.title}\n\n"
    content += f"Authors: {', '.join(a.name for a in result.authors)}\n\n"
    content += f"Published: {result.published}\n\n"
    content += f"Updated: {result.updated}\n\n"
    content += f"Categories: {', '.join(result.categories)}\n\n"
    content += f"Primary Category: {result.primary_category}\n\n"
    if result.doi:          content += f"DOI: {result.doi}\n\n"
    if result.journal_ref:  content += f"Journal Reference: {result.journal_ref}\n\n"
    if result.comment:      content += f"Comment: {result.comment}\n\n"
    content += f"=== {label.upper()} PAPER ===\n\nAbstract:\n{result.summary}"

    safe_title = re.sub(r'[^\w\s-]', '', result.title)[:100].strip().replace(' ', '_')
    filename   = secure_filename(
        f"{current_user.id}_{datetime.now().timestamp()}_healthcare_software_{safe_title}.txt"
    )
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    dataset = Dataset(
        name=f"arXiv ({label}): {result.title[:100]}",
        domain='healthcare_software',
        filename=filename,
        user_id=current_user.id,
        source_type='arxiv',
        source_url=result.entry_id
    )
    db.session.add(dataset)
    db.session.commit()
    process_dataset(dataset.id, filepath)


@app.route('/import/arxiv', methods=['POST'])
@login_required
def import_arxiv():
    """Import healthcare-software papers from arXiv (fixed filtering)."""
    search_query = request.form.get('search_query')
    max_results  = int(request.form.get('max_results', 10))

    if not search_query:
        flash('Please enter a search query')
        return redirect(url_for('dashboard'))

    try:
        query  = _build_arxiv_query(search_query)
        search = arxiv.Search(
            query=query,
            max_results=max_results * 2,   # fetch extra for secondary filter
            sort_by=arxiv.SortCriterion.Relevance
        )

        imported_count = 0

        for result in search.results():
            if not _arxiv_paper_passes(result):
                print(f"[SKIP] arXiv paper '{result.title}' — score too low")
                continue

            _save_arxiv_paper(result)
            imported_count += 1
            if imported_count >= max_results:
                break

        if imported_count > 0:
            flash(f'Successfully imported {imported_count} healthcare-software arXiv papers')
        else:
            flash(
                f'No qualifying papers found for "{search_query}". '
                f'Try "clinical NLP EHR", "medical image deep learning", '
                f'or "health informatics prediction".'
            )

    except Exception as e:
        flash(f'Error importing arXiv papers: {str(e)}')

    return redirect(url_for('dashboard'))


@app.route('/import/arxiv/advanced', methods=['POST'])
@login_required
def import_arxiv_advanced():
    """Advanced arXiv import with healthcare-software focus (fixed filtering)."""
    query       = request.form.get('query')
    category    = request.form.get('category', 'all')
    max_results = int(request.form.get('max_results', 5))
    sort_by     = request.form.get('sort_by', 'relevance')

    if not query:
        flash('Please enter a search query')
        return redirect(url_for('dashboard'))

    try:
        sort_map = {
            'relevance':       arxiv.SortCriterion.Relevance,
            'submittedDate':   arxiv.SortCriterion.SubmittedDate,
            'lastUpdatedDate': arxiv.SortCriterion.LastUpdatedDate,
        }

        search = arxiv.Search(
            query=_build_arxiv_query(query, category),
            max_results=max_results,
            sort_by=sort_map.get(sort_by, arxiv.SortCriterion.Relevance)
        )

        imported_count = 0

        for result in search.results():
            if not _arxiv_paper_passes(result):
                print(f"[SKIP] arXiv paper '{result.title}' — score too low")
                continue

            _save_arxiv_paper(result, label='Healthcare-Software Research')
            imported_count += 1

        if imported_count > 0:
            flash(f'Successfully imported {imported_count} healthcare-software arXiv papers')
        else:
            flash(
                'No qualifying papers found. '
                'Try "EHR deep learning", "clinical decision support ML", '
                'or "biomedical NLP".'
            )

    except Exception as e:
        flash(f'Error in advanced arXiv import: {str(e)}')

    return redirect(url_for('dashboard'))
# ============================================
# 6. MULTI-FILE UPLOAD ROUTE
# ============================================
@app.route('/upload_multi', methods=['GET', 'POST'])
@login_required
def upload_multi():
    """Handle multiple file uploads for cross-domain knowledge graph"""
    if request.method == 'POST':
        files = request.files.getlist('files')
        domains = request.form.getlist('domains')
        
        # Validate at least 2 files
        if len(files) < 2:
            flash('Please upload at least 2 files from different domains', 'error')
            return redirect(request.url)
        
        # Check if domains are different
        if len(set(domains)) < 2:
            flash('Please select different domains for cross-domain processing', 'error')
            return redirect(request.url)
        
        dataset_ids = []
        uploaded_files = []
        
        for i, file in enumerate(files):
            if file and file.filename:
                # Validate file extension
                if not file.filename.endswith(('.txt', '.csv')):
                    flash(f'File {file.filename} must be .txt or .csv', 'error')
                    return redirect(request.url)
                
                # Secure filename and save
                timestamp = datetime.now().timestamp()
                filename = secure_filename(f"{current_user.id}_{timestamp}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_files.append(filepath)
                
                # Create dataset record
                dataset = Dataset(
                    name=file.filename,
                    domain=domains[i],
                    filename=filename,
                    user_id=current_user.id,
                    source_type='upload'
                )
                db.session.add(dataset)
                db.session.flush()
                dataset_ids.append(dataset.id)
        
        # Commit to get dataset IDs
        db.session.commit()
        
        # Process all datasets and find cross-domain relations
        try:
            process_cross_domain_datasets(dataset_ids, uploaded_files)
            flash(f'Successfully uploaded {len(files)} datasets! Cross-domain processing started.', 'success')
        except Exception as e:
            flash(f'Error processing datasets: {str(e)}', 'error')
            print(f"Cross-domain processing error: {e}")
        
        return redirect(url_for('dashboard'))
    
    return render_template('upload_multi.html')

# ============================================
# 7. OTHER ROUTES (graph, search, admin, etc.)
# ============================================

@app.route('/graph/<int:dataset_id>')
@login_required
def view_graph(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != current_user.id and not current_user.is_admin:
        flash('Access denied')
        return redirect(url_for('dashboard'))
    
    return render_template('graph.html', dataset=dataset)

@app.route('/api/graph/<int:dataset_id>')
@login_required
def get_graph_data(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    
    entities = Entity.query.filter_by(dataset_id=dataset_id).all()
    relations = Relation.query.filter_by(dataset_id=dataset_id).all()
    
    nodes = []
    for entity in entities:
        nodes.append({
            'id': entity.id,
            'label': entity.name,
            'type': entity.type,
            'confidence': entity.confidence
        })
    
    edges = []
    for relation in relations:
        edges.append({
            'from': relation.entity1_id,
            'to': relation.entity2_id,
            'label': relation.relation_type,
            'confidence': relation.confidence,
            'approved': relation.approved
        })
    
    return jsonify({'nodes': nodes, 'edges': edges})

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    if request.method == 'POST':
        query = request.form.get('query')
        dataset_id = request.form.get('dataset_id')
        
        if dataset_id:
            dataset = Dataset.query.get(dataset_id)
            if dataset.user_id != current_user.id and not current_user.is_admin:
                return jsonify({'error': 'Access denied'})
            
            # Get all entities and relations for the dataset
            entities = Entity.query.filter_by(dataset_id=dataset_id).all()
            relations = Relation.query.filter_by(dataset_id=dataset_id).all()
            
            # Perform semantic search
            results = semantic_search(query, entities, relations, encoder)
            
            return jsonify(results)
    
    datasets = Dataset.query.filter_by(user_id=current_user.id, processed=True).all()
    return render_template('search.html', datasets=datasets)

@app.route('/admin')
@login_required
def admin():
    if not current_user.is_admin:
        flash('Access denied')
        return redirect(url_for('dashboard'))
    
    users = User.query.all()
    datasets = Dataset.query.all()
    entities = Entity.query.all()
    relations = Relation.query.all()
    
    # Count by source type
    source_stats = {
        'upload': Dataset.query.filter_by(source_type='upload').count(),
        'wikipedia': Dataset.query.filter_by(source_type='wikipedia').count(),
        'arxiv': Dataset.query.filter_by(source_type='arxiv').count()
    }
    
    stats = {
        'total_users': len(users),
        'total_datasets': len(datasets),
        'total_entities': len(entities),
        'total_relations': len(relations),
        'pending_relations': Relation.query.filter_by(approved=False).count(),
        'source_stats': source_stats
    }
    
    return render_template('admin.html', users=users, datasets=datasets, 
                         entities=entities, relations=relations, stats=stats)

@app.route('/api/dataset_stats/<int:dataset_id>')
@login_required
def dataset_stats(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != current_user.id and not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    # Get entity types
    entity_types = {}
    for entity in dataset.entities:
        entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
    
    # Get relation types
    relation_types = {}
    for relation in dataset.relations:
        relation_types[relation.relation_type] = relation_types.get(relation.relation_type, 0) + 1
    
    return jsonify({
        'entity_types': entity_types,
        'relation_types': relation_types,
        'source_type': dataset.source_type,
        'source_url': dataset.source_url
    })

@app.route('/api/dataset/<int:dataset_id>', methods=['DELETE'])
@login_required
def delete_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != current_user.id and not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    # Delete file
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
        if os.path.exists(filepath):
            os.remove(filepath)
    except:
        pass
    
    db.session.delete(dataset)
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/merge_entities', methods=['POST'])
@login_required
def merge_entities():
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.json
    entity1_id = data.get('entity1_id')
    entity2_id = data.get('entity2_id')
    
    entity2 = Entity.query.get(entity2_id)
    entity2.merged_with = entity1_id
    
    # Update relations
    relations = Relation.query.filter(
        (Relation.entity1_id == entity2_id) | (Relation.entity2_id == entity2_id)
    ).all()
    
    for rel in relations:
        if rel.entity1_id == entity2_id:
            rel.entity1_id = entity1_id
        if rel.entity2_id == entity2_id:
            rel.entity2_id = entity1_id
    
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/approve_relation/<int:relation_id>', methods=['POST'])
@login_required
def approve_relation(relation_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    relation = Relation.query.get(relation_id)
    relation.approved = True
    db.session.commit()
    
    return jsonify({'success': True})

# ============================================
# 8. PROCESSING FUNCTIONS
# ============================================

def process_dataset(dataset_id, filepath):
    """Process a single dataset"""
    try:
        dataset = Dataset.query.get(dataset_id)
        
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Preprocess
        clean_text = preprocess_text(text)
        
        # Extract entities
        doc = nlp(clean_text)
        entity_objects = []
        
        for ent in doc.ents:
            entity = Entity(
                name=ent.text,
                type=ent.label_,
                dataset_id=dataset_id,
                confidence=0.95
            )
            db.session.add(entity)
            db.session.flush()
            entity_objects.append(entity)
        
        # Extract relations
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ('nsubj', 'nsubjpass') and token.head.pos_ == 'VERB':
                    subject = token.text
                    verb = token.head.text
                    
                    for child in token.head.children:
                        if child.dep_ in ('dobj', 'attr', 'prep'):
                            object_text = child.text
                            
                            entity1 = find_entity_in_text(subject, entity_objects)
                            entity2 = find_entity_in_text(object_text, entity_objects)
                            
                            if entity1 and entity2:
                                relation = Relation(
                                    entity1_id=entity1.id,
                                    entity2_id=entity2.id,
                                    relation_type=verb,
                                    confidence=0.85,
                                    dataset_id=dataset_id
                                )
                                db.session.add(relation)
        
        dataset.processed = True
        db.session.commit()
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        db.session.rollback()

def find_entity_in_text(text, entities):
    """Find entity by text match"""
    text_lower = text.lower()
    for entity in entities:
        if text_lower in entity.name.lower() or entity.name.lower() in text_lower:
            return entity
    return None

def process_cross_domain_datasets(dataset_ids, filepaths):
    """Process multiple datasets and find cross-domain relationships"""
    try:
        all_entities = []
        
        # First, process each dataset individually
        for idx, dataset_id in enumerate(dataset_ids):
            dataset = Dataset.query.get(dataset_id)
            filepath = filepaths[idx]
            
            # Read file content
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Preprocess text
            clean_text = preprocess_text(text)
            
            # Extract entities using spaCy
            doc = nlp(clean_text)
            entity_objects = []
            
            for ent in doc.ents:
                entity = Entity(
                    name=ent.text,
                    type=ent.label_,
                    dataset_id=dataset_id,
                    confidence=0.95
                )
                db.session.add(entity)
                db.session.flush()
                entity_objects.append(entity)
                all_entities.append(entity)
            
            # Extract relations within the same dataset
            for sent in doc.sents:
                for token in sent:
                    if token.dep_ in ('nsubj', 'nsubjpass') and token.head.pos_ == 'VERB':
                        subject = token.text
                        verb = token.head.text
                        
                        for child in token.head.children:
                            if child.dep_ in ('dobj', 'attr', 'prep'):
                                object_text = child.text
                                
                                entity1 = find_entity_in_text(subject, entity_objects)
                                entity2 = find_entity_in_text(object_text, entity_objects)
                                
                                if entity1 and entity2:
                                    relation = Relation(
                                        entity1_id=entity1.id,
                                        entity2_id=entity2.id,
                                        relation_type=verb,
                                        confidence=0.85,
                                        dataset_id=dataset_id,
                                        approved=False
                                    )
                                    db.session.add(relation)
            
            dataset.processed = True
            db.session.commit()
        
        # Now find CROSS-DOMAIN relationships
        find_cross_domain_relations(all_entities, dataset_ids)
        
        print(f"Cross-domain processing complete for {len(dataset_ids)} datasets")
        
    except Exception as e:
        print(f"Error in cross-domain processing: {e}")
        db.session.rollback()
        raise e

def find_cross_domain_relations(all_entities, dataset_ids):
    """Find relationships between entities from different domains"""
    try:
        # Group entities by dataset
        entities_by_dataset = {}
        for entity in all_entities:
            if entity.dataset_id not in entities_by_dataset:
                entities_by_dataset[entity.dataset_id] = []
            entities_by_dataset[entity.dataset_id].append(entity)
        
        relation_count = 0
        
        # Compare entities across different datasets
        for i in range(len(dataset_ids)):
            for j in range(i + 1, len(dataset_ids)):
                dataset1_id = dataset_ids[i]
                dataset2_id = dataset_ids[j]
                
                entities1 = entities_by_dataset.get(dataset1_id, [])
                entities2 = entities_by_dataset.get(dataset2_id, [])
                
                # Look for potential cross-domain relationships
                for entity1 in entities1:
                    for entity2 in entities2:
                        # Check for semantic similarity
                        similarity = check_entity_similarity(entity1.name, entity2.name)
                        
                        if similarity > 0.7:  # High similarity - likely same concept
                            # Check if relation already exists
                            existing = Relation.query.filter(
                                ((Relation.entity1_id == entity1.id) & (Relation.entity2_id == entity2.id)) |
                                ((Relation.entity1_id == entity2.id) & (Relation.entity2_id == entity1.id))
                            ).first()
                            
                            if not existing:
                                relation = Relation(
                                    entity1_id=entity1.id,
                                    entity2_id=entity2.id,
                                    relation_type='same_as',
                                    confidence=similarity,
                                    dataset_id=dataset1_id,
                                    approved=False
                                )
                                db.session.add(relation)
                                relation_count += 1
                        
                        elif similarity > 0.4:  # Medium similarity - possible relation
                            relation_type = infer_cross_domain_relation(entity1, entity2)
                            if relation_type:
                                # Check if relation already exists
                                existing = Relation.query.filter(
                                    ((Relation.entity1_id == entity1.id) & (Relation.entity2_id == entity2.id)) |
                                    ((Relation.entity1_id == entity2.id) & (Relation.entity2_id == entity1.id))
                                ).first()
                                
                                if not existing:
                                    relation = Relation(
                                        entity1_id=entity1.id,
                                        entity2_id=entity2.id,
                                        relation_type=relation_type,
                                        confidence=similarity,
                                        dataset_id=dataset1_id,
                                        approved=False
                                    )
                                    db.session.add(relation)
                                    relation_count += 1
        
        db.session.commit()
        print(f"Created {relation_count} cross-domain relations")
        
    except Exception as e:
        print(f"Error finding cross-domain relations: {e}")
        db.session.rollback()
        raise e

def check_entity_similarity(name1, name2):
    """Check if two entity names are similar"""
    if not name1 or not name2:
        return 0.0
    
    name1 = name1.lower().strip()
    name2 = name2.lower().strip()
    
    # Exact match
    if name1 == name2:
        return 1.0
    
    # Direct string similarity
    direct_similarity = SequenceMatcher(None, name1, name2).ratio()
    
    # Check if one is substring of another
    if name1 in name2 or name2 in name1:
        substring_boost = 0.2
    else:
        substring_boost = 0
    
    # Check for word overlap
    words1 = set(name1.split())
    words2 = set(name2.split())
    if words1 and words2:
        word_overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
        word_boost = word_overlap * 0.1
    else:
        word_boost = 0
    
    final_score = min(direct_similarity + substring_boost + word_boost, 1.0)
    return final_score

def infer_cross_domain_relation(entity1, entity2):
    """Infer possible relation between entities from different domains"""
    
    # Common cross-domain relation patterns
    relation_patterns = [
        ('PERSON', 'ORG', 'works_for'),
        ('PERSON', 'GPE', 'lives_in'),
        ('ORG', 'GPE', 'located_in'),
        ('PRODUCT', 'ORG', 'produced_by'),
        ('TECHNOLOGY', 'SCIENCE', 'based_on'),
        ('DISEASE', 'MEDICINE', 'treated_by'),
        ('LAW', 'COUNTRY', 'applicable_in'),
        ('PERSON', 'PRODUCT', 'invented'),
        ('ORG', 'PRODUCT', 'develops'),
        ('SCIENCE', 'TECHNOLOGY', 'enables')
    ]
    
    for type1, type2, relation in relation_patterns:
        if (entity1.type == type1 and entity2.type == type2):
            return relation
        elif (entity1.type == type2 and entity2.type == type1):
            return reverse_relation(relation)
    
    return None

def reverse_relation(relation):
    """Get reverse of a relation"""
    reversals = {
        'works_for': 'employs',
        'lives_in': 'has_resident',
        'located_in': 'contains',
        'produced_by': 'produces',
        'based_on': 'used_in',
        'treated_by': 'treats',
        'applicable_in': 'has_law',
        'invented': 'was_invented_by',
        'develops': 'developed_by',
        'enables': 'enabled_by'
    }
    return reversals.get(relation, 'related_to')


# ============================================
@app.route('/dataset/<int:dataset_id>/view')
@login_required
def view_original_file(dataset_id):
    """View the original uploaded file in the browser"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Make sure the user owns this dataset
    if dataset.user_id != current_user.id and not current_user.is_admin:
        abort(403)
    
    # Construct the full file path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        abort(404, description="Original file not found")
    
    # For text files, display them in the browser
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith('text/'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return render_template('view_file.html', 
                                 dataset=dataset, 
                                 content=content,
                                 filename=dataset.filename)
        except UnicodeDecodeError:
            # Try different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            return render_template('view_file.html', 
                                 dataset=dataset, 
                                 content=content,
                                 filename=dataset.filename)
    else:
        # For other file types, just serve the file
        return send_file(file_path, as_attachment=False)

@app.route('/dataset/<int:dataset_id>/download')
@login_required
def download_original_file(dataset_id):
    """Download the original uploaded file"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    # Make sure the user owns this dataset
    if dataset.user_id != current_user.id and not current_user.is_admin:
        abort(403)
    
    # Construct the full file path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        abort(404, description="Original file not found")
    
    return send_file(
        file_path,
        as_attachment=True,
        download_name=dataset.name,  # Use the original display name
        mimetype=mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
    )

@app.route('/dataset/<int:dataset_id>/file-info')
@login_required
def file_info(dataset_id):
    """Get information about the uploaded file"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    if dataset.user_id != current_user.id and not current_user.is_admin:
        abort(403)
    
    # Construct the full file path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    file_stats = os.stat(file_path)
    file_size_bytes = file_stats.st_size
    
    # Format file size
    if file_size_bytes < 1024:
        file_size = f"{file_size_bytes} B"
    elif file_size_bytes < 1024 * 1024:
        file_size = f"{file_size_bytes / 1024:.1f} KB"
    elif file_size_bytes < 1024 * 1024 * 1024:
        file_size = f"{file_size_bytes / (1024 * 1024):.1f} MB"
    else:
        file_size = f"{file_size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    return jsonify({
        'filename': dataset.name,  # Display name
        'stored_filename': dataset.filename,  # Actual stored filename
        'size': file_size,
        'uploaded': dataset.uploaded_at.strftime('%Y-%m-%d %H:%M:%S'),
        'type': mimetypes.guess_type(file_path)[0] or 'Unknown',
        'source_type': dataset.source_type,
        'domain': dataset.domain
    })

# Also add a route to list all files in a dataset (useful for debugging)
@app.route('/dataset/<int:dataset_id>/files')
@login_required
def list_dataset_files(dataset_id):
    """List all files associated with a dataset (for debugging)"""
    dataset = Dataset.query.get_or_404(dataset_id)
    
    if dataset.user_id != current_user.id and not current_user.is_admin:
        abort(403)
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset.filename)
    
    file_info = {
        'dataset_id': dataset.id,
        'dataset_name': dataset.name,
        'stored_filename': dataset.filename,
        'file_exists': os.path.exists(file_path),
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'full_path': file_path,
        'source_type': dataset.source_type,
        'source_url': dataset.source_url
    }
    
    if os.path.exists(file_path):
        file_info['file_size'] = os.path.getsize(file_path)
        file_info['modified'] = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
    
    return jsonify(file_info)



# 9. RUN THE APPLICATION
# ============================================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # Create admin user if not exists
        if not User.query.filter_by(email='admin@example.com').first():
            admin = User(
                username='admin',
                email='admin@example.com',
                password_hash=generate_password_hash('admin123'),
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
    
    app.run(debug=True)