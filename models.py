from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

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
    # New columns for external sources
    source_type = db.Column(db.String(50), default='upload')  # 'upload', 'wikipedia', 'arxiv'
    source_url = db.Column(db.String(500), nullable=True)  # Original source URL
    
    # Relationships
    entities = db.relationship('Entity', backref='dataset', lazy=True, cascade='all, delete-orphan')
    relations = db.relationship('Relation', backref='dataset', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Dataset {self.name}>'
    
    def get_stats(self):
        return {
            'entity_count': len(self.entities),
            'relation_count': len(self.relations),
            'entity_types': self.get_entity_types(),
            'source_type': self.source_type,
            'source_url': self.source_url
        }
    
    def get_entity_types(self):
        types = {}
        for entity in self.entities:
            types[entity.type] = types.get(entity.type, 0) + 1
        return types

class Entity(db.Model):
    __tablename__ = 'entities'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    confidence = db.Column(db.Float, default=1.0)
    merged_with = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    relations_from = db.relationship('Relation', 
                                    foreign_keys='Relation.entity1_id',
                                    backref='entity1_ref',
                                    lazy=True,
                                    cascade='all, delete-orphan')
    relations_to = db.relationship('Relation',
                                  foreign_keys='Relation.entity2_id',
                                  backref='entity2_ref',
                                  lazy=True,
                                  cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Entity {self.name} ({self.type})>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'confidence': self.confidence,
            'dataset_id': self.dataset_id
        }

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
    entity1 = db.relationship('Entity', foreign_keys=[entity1_id], overlaps="relations_from,entity1_ref")
    entity2 = db.relationship('Entity', foreign_keys=[entity2_id], overlaps="relations_to,entity2_ref")
    
    def __repr__(self):
        return f'<Relation {self.entity1.name} - {self.relation_type} - {self.entity2.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'entity1': self.entity1.name if self.entity1 else None,
            'entity2': self.entity2.name if self.entity2 else None,
            'entity1_id': self.entity1_id,
            'entity2_id': self.entity2_id,
            'relation_type': self.relation_type,
            'confidence': self.confidence,
            'approved': self.approved,
            'dataset_id': self.dataset_id
        }

class ProcessingJob(db.Model):
    __tablename__ = 'processing_jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, processing, completed, failed
    progress = db.Column(db.Integer, default=0)
    error_message = db.Column(db.Text, nullable=True)
    started_at = db.Column(db.DateTime, nullable=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    dataset = db.relationship('Dataset', backref='processing_jobs')
    
    def __repr__(self):
        return f'<ProcessingJob {self.dataset_id} - {self.status}>'

class Feedback(db.Model):
    __tablename__ = 'feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    relation_id = db.Column(db.Integer, db.ForeignKey('relations.id'), nullable=True)
    feedback_type = db.Column(db.String(20), nullable=False)  # correct, incorrect, suggestion
    comment = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref='feedback')
    relation = db.relationship('Relation', backref='feedback')
    
    def __repr__(self):
        return f'<Feedback {self.feedback_type} by {self.user.username}>'