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
    return redirect(url_for('login'))

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
    return redirect(url_for('login'))

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
    
    stats = {
        'total_datasets': len(datasets),
        'total_entities': Entity.query.join(Dataset).filter(Dataset.user_id == current_user.id).count(),
        'total_relations': Relation.query.join(Dataset).filter(Dataset.user_id == current_user.id).count(),
        'cross_domain_datasets': cross_domain_datasets,
        'cross_domain_relations': cross_domain_relations
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
                user_id=current_user.id
            )
            db.session.add(dataset)
            db.session.commit()
            
            # Process the dataset
            process_dataset(dataset.id, filepath)
            
            flash('Dataset uploaded and processing started!')
            return redirect(url_for('dashboard'))
    
    return render_template('upload.html')

# ============================================
# 5. MULTI-FILE UPLOAD ROUTE (NOW app IS DEFINED)
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
                    user_id=current_user.id
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
# 6. OTHER ROUTES (graph, search, admin, etc.)
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
    
    stats = {
        'total_users': len(users),
        'total_datasets': len(datasets),
        'total_entities': len(entities),
        'total_relations': len(relations),
        'pending_relations': Relation.query.filter_by(approved=False).count()
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
        'relation_types': relation_types
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
# 7. PROCESSING FUNCTIONS
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
# 8. RUN THE APPLICATION
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