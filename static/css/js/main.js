// Main JavaScript file for common functionality

// Global variables
let currentUser = null;
let toastContainer = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeToasts();
    initializeTooltips();
    initializeFileUploads();
    initializeCharts();
    setupAjaxCSRF();
});

// Toast notification system
function initializeToasts() {
    toastContainer = document.createElement('div');
    toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
    toastContainer.style.zIndex = '11';
    document.body.appendChild(toastContainer);
}

function showToast(message, type = 'info', duration = 3000) {
    const toastId = 'toast-' + Date.now();
    const bgColor = type === 'success' ? 'bg-success' : 
                   type === 'error' ? 'bg-danger' : 
                   type === 'warning' ? 'bg-warning' : 'bg-info';
    
    const toastHtml = `
        <div id="${toastId}" class="toast ${bgColor} text-white" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header ${bgColor} text-white">
                <strong class="me-auto">${type.charAt(0).toUpperCase() + type.slice(1)}</strong>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, { autohide: true, delay: duration });
    toast.show();
    
    // Remove toast after it's hidden
    toastElement.addEventListener('hidden.bs.toast', function() {
        this.remove();
    });
}

// Initialize Bootstrap tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// File upload preview and validation
function initializeFileUploads() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const file = this.files[0];
            if (!file) return;
            
            // Validate file size (16MB max)
            const maxSize = 16 * 1024 * 1024; // 16MB in bytes
            if (file.size > maxSize) {
                showToast('File size must be less than 16MB', 'error');
                this.value = '';
                return;
            }
            
            // Validate file type
            const validTypes = ['.txt', '.csv'];
            const fileExt = '.' + file.name.split('.').pop().toLowerCase();
            if (!validTypes.includes(fileExt)) {
                showToast('Please upload a .txt or .csv file', 'error');
                this.value = '';
                return;
            }
            
            // Show file preview
            const preview = document.createElement('div');
            preview.className = 'file-preview mt-2 p-2 bg-light rounded';
            preview.innerHTML = `
                <small>
                    <i class="bi bi-file-text"></i> ${file.name} 
                    (${(file.size / 1024).toFixed(2)} KB)
                </small>
            `;
            
            // Remove existing preview
            const existingPreview = this.parentElement.querySelector('.file-preview');
            if (existingPreview) {
                existingPreview.remove();
            }
            
            this.parentElement.appendChild(preview);
        });
    });
}

// Chart initialization
function initializeCharts() {
    // Check if Chart.js is available
    if (typeof Chart === 'undefined') return;
    
    // Entity type distribution chart
    const entityChart = document.getElementById('entityTypeChart');
    if (entityChart) {
        fetch('/api/entity-stats')
            .then(response => response.json())
            .then(data => {
                new Chart(entityChart, {
                    type: 'doughnut',
                    data: {
                        labels: data.labels,
                        datasets: [{
                            data: data.values,
                            backgroundColor: [
                                '#ff6b6b',
                                '#4ecdc4',
                                '#45b7d1',
                                '#96ceb4',
                                '#feca57',
                                '#ff9ff3'
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                position: 'bottom'
                            }
                        }
                    }
                });
            });
    }
    
    // Dataset activity chart
    const activityChart = document.getElementById('activityChart');
    if (activityChart) {
        fetch('/api/activity-stats')
            .then(response => response.json())
            .then(data => {
                new Chart(activityChart, {
                    type: 'line',
                    data: {
                        labels: data.labels,
                        datasets: [{
                            label: 'Datasets Uploaded',
                            data: data.values,
                            borderColor: '#4361ee',
                            backgroundColor: 'rgba(67, 97, 238, 0.1)',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: {
                                display: false
                            }
                        }
                    }
                });
            });
    }
}

// Setup AJAX CSRF protection
function setupAjaxCSRF() {
    const csrfToken = document.querySelector('meta[name="csrf-token"]')?.getAttribute('content');
    
    if (csrfToken) {
        // Add CSRF token to all AJAX requests
        $.ajaxSetup({
            beforeSend: function(xhr, settings) {
                if (!/^(GET|HEAD|OPTIONS|TRACE)$/i.test(settings.type) && !this.crossDomain) {
                    xhr.setRequestHeader('X-CSRFToken', csrfToken);
                }
            }
        });
    }
}

// Form validation
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return true;
    
    let isValid = true;
    const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
    
    inputs.forEach(input => {
        if (!input.value.trim()) {
            input.classList.add('is-invalid');
            isValid = false;
            
            // Add error message if not exists
            let errorDiv = input.nextElementSibling;
            if (!errorDiv || !errorDiv.classList.contains('invalid-feedback')) {
                errorDiv = document.createElement('div');
                errorDiv.className = 'invalid-feedback';
                errorDiv.textContent = 'This field is required';
                input.parentNode.insertBefore(errorDiv, input.nextSibling);
            }
        } else {
            input.classList.remove('is-invalid');
        }
    });
    
    return isValid;
}

// Loading spinner
function showLoading(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = `
        <div class="text-center p-5">
            <div class="spinner mb-3"></div>
            <p class="text-muted">Loading...</p>
        </div>
    `;
}

function hideLoading(containerId, content) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    if (content) {
        container.innerHTML = content;
    }
}

// Data export
function exportData(data, filename, type = 'json') {
    let blob;
    let url;
    
    if (type === 'json') {
        blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    } else if (type === 'csv') {
        // Convert to CSV
        const csv = convertToCSV(data);
        blob = new Blob([csv], { type: 'text/csv' });
    }
    
    url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}.${type}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    
    showToast(`Data exported as ${filename}.${type}`, 'success');
}

function convertToCSV(data) {
    if (!data || !data.length) return '';
    
    const headers = Object.keys(data[0]);
    const rows = data.map(item => headers.map(header => JSON.stringify(item[header] || '')).join(','));
    
    return [headers.join(','), ...rows].join('\n');
}

// Search functionality
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Auto-resize textarea
function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

document.querySelectorAll('textarea').forEach(textarea => {
    textarea.addEventListener('input', function() {
        autoResizeTextarea(this);
    });
});

// Copy to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard!', 'success');
    }).catch(() => {
        showToast('Failed to copy', 'error');
    });
}

// Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Check if element is in viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

// Smooth scroll to element
function smoothScrollTo(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// Dark mode toggle (optional)
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    const isDark = document.body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDark);
    
    showToast(`Dark mode ${isDark ? 'enabled' : 'disabled'}`, 'info');
}

// Check for saved dark mode preference
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
}