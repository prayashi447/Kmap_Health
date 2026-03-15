// Knowledge Graph Visualization with vis.js

class KnowledgeGraph {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error('Container not found:', containerId);
            return;
        }
        
        this.options = {
            physics: options.physics !== false,
            interactive: options.interactive !== false,
            height: options.height || '600px',
            ...options
        };
        
        this.network = null;
        this.nodes = new vis.DataSet([]);
        this.edges = new vis.DataSet([]);
        this.currentDataset = null;
        
        this.init();
    }
    
    init() {
        // Set container height
        this.container.style.height = this.options.height;
        
        // Initialize network data
        const data = {
            nodes: this.nodes,
            edges: this.edges
        };
        
        // Network options
        const networkOptions = {
            nodes: {
                shape: 'dot',
                size: 25,
                font: {
                    size: 14,
                    color: '#333'
                },
                borderWidth: 2,
                shadow: true
            },
            edges: {
                width: 2,
                shadow: true,
                font: {
                    size: 12,
                    align: 'middle'
                },
                arrows: {
                    to: {
                        enabled: true,
                        scaleFactor: 0.5
                    }
                },
                smooth: {
                    type: 'continuous'
                }
            },
            physics: {
                enabled: this.options.physics,
                stabilization: {
                    enabled: true,
                    iterations: 1000
                },
                barnesHut: {
                    gravitationalConstant: -3000,
                    centralGravity: 0.3,
                    springLength: 95,
                    springConstant: 0.04,
                    damping: 0.09
                }
            },
            interaction: {
                hover: true,
                tooltipDelay: 200,
                hideEdgesOnDrag: true,
                navigationButtons: true,
                keyboard: true
            },
            layout: {
                improvedLayout: true,
                hierarchical: {
                    enabled: false
                }
            },
            groups: {
                PERSON: {
                    color: '#ff6b6b',
                    shape: 'dot'
                },
                ORG: {
                    color: '#4ecdc4',
                    shape: 'dot'
                },
                GPE: {
                    color: '#45b7d1',
                    shape: 'dot'
                },
                LOC: {
                    color: '#45b7d1',
                    shape: 'dot'
                },
                DATE: {
                    color: '#96ceb4',
                    shape: 'dot'
                },
                MONEY: {
                    color: '#feca57',
                    shape: 'dot'
                },
                PERCENT: {
                    color: '#ff9ff3',
                    shape: 'dot'
                },
                default: {
                    color: '#96ceb4',
                    shape: 'dot'
                }
            }
        };
        
        // Initialize network
        this.network = new vis.Network(this.container, data, networkOptions);
        
        // Setup event listeners
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Node click event
        this.network.on('click', (params) => {
            if (params.nodes.length > 0) {
                this.onNodeClick(params.nodes[0]);
            }
        });
        
        // Node hover event
        this.network.on('hoverNode', (params) => {
            this.onNodeHover(params.node);
        });
        
        // Edge click event
        this.network.on('click', (params) => {
            if (params.edges.length > 0) {
                this.onEdgeClick(params.edges[0]);
            }
        });
        
        // Double click event
        this.network.on('doubleClick', (params) => {
            if (params.nodes.length > 0) {
                this.onNodeDoubleClick(params.nodes[0]);
            }
        });
        
        // Stabilization events
        this.network.on('stabilizationProgress', (params) => {
            this.onStabilizationProgress(params);
        });
        
        this.network.on('stabilized', () => {
            this.onStabilized();
        });
        
        // Drag events
        this.network.on('dragStart', () => {
            this.container.style.cursor = 'grabbing';
        });
        
        this.network.on('dragEnd', () => {
            this.container.style.cursor = 'default';
        });
    }
    
    loadGraphData(datasetId) {
        this.currentDataset = datasetId;
        
        // Show loading
        this.container.innerHTML = '<div class="text-center p-5"><div class="spinner"></div><p>Loading graph...</p></div>';
        
        // Fetch graph data
        fetch(`/api/graph/${datasetId}`)
            .then(response => response.json())
            .then(data => {
                this.clear();
                this.addNodes(data.nodes);
                this.addEdges(data.edges);
                this.container.innerHTML = ''; // Clear loading
                this.network.redraw();
                
                // Fit to view
                this.network.fit();
                
                // Show success message
                window.showToast?.('Graph loaded successfully', 'success');
            })
            .catch(error => {
                console.error('Error loading graph:', error);
                this.container.innerHTML = '<div class="alert alert-danger">Failed to load graph</div>';
                window.showToast?.('Failed to load graph', 'error');
            });
    }
    
    addNodes(nodes) {
        nodes.forEach(node => {
            this.nodes.add({
                id: node.id,
                label: node.label,
                title: this.createNodeTooltip(node),
                group: node.type || 'default',
                value: node.confidence || 1,
                font: {
                    size: 14 + (node.confidence ? node.confidence * 10 : 0)
                }
            });
        });
    }
    
    addEdges(edges) {
        edges.forEach(edge => {
            this.edges.add({
                id: edge.id || `${edge.from}-${edge.to}`,
                from: edge.from,
                to: edge.to,
                label: edge.label,
                title: `Confidence: ${(edge.confidence * 100).toFixed(1)}%`,
                color: edge.approved ? '#28a745' : '#dc3545',
                width: 2 + (edge.confidence || 0.5) * 2,
                dashes: edge.approved ? false : true,
                arrows: 'to'
            });
        });
    }
    
    createNodeTooltip(node) {
        return `
            <div style="padding: 10px;">
                <strong>${node.label}</strong><br>
                Type: ${node.type}<br>
                Confidence: ${(node.confidence * 100).toFixed(1)}%<br>
                ID: ${node.id}
            </div>
        `;
    }
    
    onNodeClick(nodeId) {
        const node = this.nodes.get(nodeId);
        if (!node) return;
        
        // Highlight connected nodes
        const connectedNodes = this.network.getConnectedNodes(nodeId);
        const connectedEdges = this.network.getConnectedEdges(nodeId);
        
        // Update node info panel
        const infoPanel = document.getElementById('nodeInfo');
        if (infoPanel) {
            infoPanel.innerHTML = `
                <h5>${node.label}</h5>
                <p><strong>Type:</strong> ${node.group}</p>
                <p><strong>Connected to:</strong> ${connectedNodes.length} nodes</p>
                <p><strong>Relations:</strong> ${connectedEdges.length}</p>
                <button class="btn btn-sm btn-primary mt-2" onclick="graph.focusOnNode(${nodeId})">
                    Focus
                </button>
            `;
            infoPanel.style.display = 'block';
        }
        
        // Emit custom event
        const event = new CustomEvent('graphNodeClick', { 
            detail: { nodeId, node, connectedNodes, connectedEdges } 
        });
        this.container.dispatchEvent(event);
    }
    
    onNodeHover(nodeId) {
        // Show tooltip with node info
        const node = this.nodes.get(nodeId);
        if (node && node.title) {
            // Tooltip is handled by vis.js
        }
    }
    
    onEdgeClick(edgeId) {
        const edge = this.edges.get(edgeId);
        if (!edge) return;
        
        // Emit custom event
        const event = new CustomEvent('graphEdgeClick', { detail: { edgeId, edge } });
        this.container.dispatchEvent(event);
        
        // Show edge details
        window.showToast?.(`Relation: ${edge.label}`, 'info');
    }
    
    onNodeDoubleClick(nodeId) {
        // Zoom to node
        this.focusOnNode(nodeId);
        
        // Expand node (show more connections)
        this.expandNode(nodeId);
    }
    
    focusOnNode(nodeId) {
        this.network.focus(nodeId, {
            scale: 1.5,
            animation: true
        });
    }
    
    expandNode(nodeId, depth = 1) {
        // Fetch subgraph data
        fetch(`/api/subgraph/${this.currentDataset}/${nodeId}?depth=${depth}`)
            .then(response => response.json())
            .then(data => {
                this.addNodes(data.nodes);
                this.addEdges(data.edges);
                this.network.redraw();
            })
            .catch(error => console.error('Error expanding node:', error));
    }
    
    onStabilizationProgress(params) {
        const progressDiv = document.getElementById('graphProgress');
        if (progressDiv) {
            progressDiv.style.width = Math.round(params.percentage) + '%';
        }
    }
    
    onStabilized() {
        const progressDiv = document.getElementById('graphProgress');
        if (progressDiv) {
            progressDiv.style.display = 'none';
        }
    }
    
    clear() {
        this.nodes.clear();
        this.edges.clear();
    }
    
    filterNodes(filterFn) {
        const allNodes = this.nodes.get();
        const filteredIds = allNodes.filter(filterFn).map(n => n.id);
        
        this.network.setSelection({ nodes: filteredIds });
        
        // Hide unfiltered nodes
        this.network.setOptions({
            nodes: {
                hidden: node => !filteredIds.includes(node.id)
            }
        });
    }
    
    resetFilter() {
        this.network.setOptions({
            nodes: {
                hidden: false
            }
        });
        this.network.fit();
    }
    
    exportGraph(format = 'png') {
        if (format === 'png') {
            const canvas = this.container.querySelector('canvas');
            if (canvas) {
                const link = document.createElement('a');
                link.download = `graph_${Date.now()}.png`;
                link.href = canvas.toDataURL();
                link.click();
            }
        } else if (format === 'json') {
            const data = {
                nodes: this.nodes.get(),
                edges: this.edges.get()
            };
            window.exportData?.(data, `graph_${Date.now()}`, 'json');
        }
    }
    
    zoomIn() {
        const scale = this.network.getScale() * 1.2;
        this.network.moveTo({ scale });
    }
    
    zoomOut() {
        const scale = this.network.getScale() * 0.8;
        this.network.moveTo({ scale });
    }
    
    resetView() {
        this.network.fit();
    }
    
    togglePhysics() {
        const currentState = this.network.getOptions().physics.enabled;
        this.network.setOptions({ physics: { enabled: !currentState } });
    }
    
    searchNodes(query) {
        const allNodes = this.nodes.get();
        const results = allNodes.filter(node => 
            node.label.toLowerCase().includes(query.toLowerCase())
        );
        
        if (results.length > 0) {
            this.network.focus(results[0].id);
            this.highlightNodes(results.map(r => r.id));
        }
        
        return results;
    }
    
    highlightNodes(nodeIds, color = '#ff6b6b') {
        // Reset colors
        const allNodes = this.nodes.get();
        allNodes.forEach(node => {
            this.nodes.update({
                id: node.id,
                color: undefined // Use default group color
            });
        });
        
        // Highlight selected nodes
        nodeIds.forEach(id => {
            this.nodes.update({
                id: id,
                color: {
                    background: color,
                    border: '#333'
                }
            });
        });
    }
}

// Initialize graph when page loads
document.addEventListener('DOMContentLoaded', function() {
    const graphContainer = document.getElementById('graph-container');
    if (graphContainer && window.initializeGraph) {
        window.graph = new KnowledgeGraph('graph-container', {
            physics: true,
            height: '600px'
        });
        
        // Load graph if dataset ID is available
        const datasetId = graphContainer.dataset.datasetId;
        if (datasetId) {
            window.graph.loadGraphData(datasetId);
        }
    }
});

// Export for use in other scripts
window.KnowledgeGraph = KnowledgeGraph;