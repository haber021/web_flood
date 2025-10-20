document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('alert-form');
    const titleInput = document.getElementById('id_title');
    const descriptionInput = document.getElementById('id_description');
    const severitySelect = document.getElementById('id_severity_level');
    const predictedTimeInput = document.getElementById('id_predicted_flood_time');
    const scheduledTimeInput = document.getElementById('id_scheduled_send_time');
    const affectedBarangays = document.getElementById('id_affected_barangays');
    const previewBtn = document.getElementById('preview-alert-btn');
    const alertPreview = document.getElementById('alert-preview');
    const recipientCount = document.getElementById('recipient-count');
    const recipientBadge = document.getElementById('recipient-badge');
    const totalRecipients = document.getElementById('total-recipients');
    const selectedBarangayCount = document.getElementById('selected-barangay-count');

    // Initialize flatpickr for datetime inputs
    if (typeof flatpickr !== 'undefined') {
        if (predictedTimeInput) {
            flatpickr("#id_predicted_flood_time", {
                enableTime: true,
                dateFormat: "Y-m-d H:i",
                minDate: 'today',
                time_24hr: true,
                allowInput: true
            });
        }

        if (scheduledTimeInput) {
            flatpickr("#id_scheduled_send_time", {
                enableTime: true,
                dateFormat: "Y-m-d H:i",
                minDate: 'today',
                time_24hr: true,
                allowInput: true
            });
        }
    }

    // Update recipient count when barangay selection changes
    if (affectedBarangays) {
        affectedBarangays.addEventListener('change', function() {
            updateRecipientCount();
            updateAlertPreview();
        });
    }

    // Update preview when form fields change
    const formElements = [titleInput, descriptionInput, severitySelect, predictedTimeInput, scheduledTimeInput];
    formElements.forEach(element => {
        if (element) {
            element.addEventListener('input', updateAlertPreview);
            element.addEventListener('change', updateAlertPreview);
        }
    });

    // Preview button click handler
    if (previewBtn && alertPreview) {
        previewBtn.addEventListener('click', function(e) {
            e.preventDefault();
            updateAlertPreview();
            alertPreview.scrollIntoView({ behavior: 'smooth' });
        });
    }

    // Form submission handler
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Validate form
            if (!form.checkValidity()) {
                e.stopPropagation();
                form.classList.add('was-validated');
                
                // Scroll to first invalid field
                const firstInvalid = form.querySelector(':invalid');
                if (firstInvalid) {
                    firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    firstInvalid.focus();
                }
                return;
            }

            // If a barangay multi-select exists, enforce at least one selection.
            if (affectedBarangays) {
                const selectedBarangays = Array.from(affectedBarangays.querySelectorAll('input[type="checkbox"]:checked'));
                if (selectedBarangays.length === 0) {
                    alert('Please select at least one barangay to send the alert to.');
                    affectedBarangays.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    return;
                }

                // Show confirmation dialog with selection count
                const confirmed = confirm(`Are you sure you want to send this emergency alert to ${selectedBarangays.length} barangay(s)?`);
                if (!confirmed) {
                    return;
                }
            }

            // Show loading state
            const submitButton = form.querySelector('button[type="submit"]');
            const originalText = submitButton ? submitButton.innerHTML : null;
            if (submitButton) {
                submitButton.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i> Sending Alert...';
                submitButton.disabled = true;
            }

            // Submit form
            form.submit();
        }, false);
    }

    // Update the alert preview
    function updateAlertPreview() {
        const title = titleInput ? titleInput.value : '[Alert Title]';
        const description = descriptionInput ? descriptionInput.value : '[Alert description will appear here]';
        const severity = severitySelect && severitySelect.selectedIndex >= 0 ? 
            severitySelect.options[severitySelect.selectedIndex].text : 'High';
        
        let time = 'Now';
        if (predictedTimeInput && predictedTimeInput.value) {
            try {
                time = new Date(predictedTimeInput.value).toLocaleString();
            } catch (e) {
                console.error('Error parsing date:', e);
                time = 'Invalid date';
            }
        }
        
        // Update preview elements
        const previewTitle = document.getElementById('preview-title');
        const previewDescription = document.getElementById('preview-description');
        const previewSeverity = document.getElementById('preview-severity');
        const previewTime = document.getElementById('preview-time');
        
        if (previewTitle) previewTitle.textContent = title || '[Alert Title]';
        if (previewDescription) previewDescription.textContent = description || '[Alert description will appear here]';
        if (previewSeverity) {
            previewSeverity.textContent = severity;
            previewSeverity.className = 'badge ms-2 ' + getSeverityClass(severity);
        }
        if (previewTime) previewTime.textContent = `Time: ${time}`;
        
        // Update location based on selected barangays
        updateLocationPreview();
    }

    // Update location preview based on selected barangays
    function updateLocationPreview() {
        const previewLocation = document.getElementById('preview-location');
        if (!previewLocation || !affectedBarangays) return;

        const selectedBarangays = Array.from(affectedBarangays.querySelectorAll('input[type="checkbox"]:checked'));

        let locationText = 'Location: ';

        if (selectedBarangays.length === 0) {
            locationText += 'No areas selected';
        } else if (selectedBarangays.length === 1) {
            const label = document.querySelector(`label[for="${selectedBarangays[0].id}"]`);
            locationText += label ? label.textContent.trim() : '1 location';
        } else {
            const totalBarangays = Array.from(affectedBarangays.querySelectorAll('input[type="checkbox"]')).length;

            if (selectedBarangays.length === totalBarangays) {
                locationText += 'All Areas';
            } else {
                locationText += `${selectedBarangays.length} locations`;
            }
        }

        previewLocation.textContent = locationText;
    }

    // Update recipient count based on selected barangays
    function updateRecipientCount() {
        if (!affectedBarangays) return;

        const selectedBarangays = Array.from(affectedBarangays.querySelectorAll('input[type="checkbox"]:checked'));

        const totalBarangays = Array.from(affectedBarangays.querySelectorAll('input[type="checkbox"]')).length;
        
        // Update selected barangay count
        if (selectedBarangayCount) {
            selectedBarangayCount.textContent = selectedBarangays.length;
        }
        
        // Simulate recipient count (in a real app, this would be an API call)
        const recipientCountValue = selectedBarangays.length * 1000; // Example: 1000 recipients per barangay
        
        // Update recipient count display
        if (recipientCount) recipientCount.textContent = `${recipientCountValue.toLocaleString()} Recipients`;
        if (recipientBadge) recipientBadge.textContent = recipientCountValue.toLocaleString();
        if (totalRecipients) totalRecipients.textContent = `${recipientCountValue.toLocaleString()} recipients`;
        
    }

    // Get appropriate Bootstrap class based on severity level
    function getSeverityClass(severity) {
        if (!severity) return 'bg-secondary';
        
        switch(severity.toLowerCase()) {
            case 'high':
            case 'critical':
                return 'bg-danger';
            case 'medium':
            case 'moderate':
                return 'bg-warning text-dark';
            case 'low':
            case 'minor':
                return 'bg-info';
            default:
                return 'bg-secondary';
        }
    }

    // Initialize
    updateAlertPreview();
    updateRecipientCount();

    // Add event listener for help button
    const helpButton = document.querySelector('[data-bs-target="#helpModal"]');
    if (helpButton) {
        helpButton.addEventListener('click', function() {
            const helpModalElement = document.getElementById('helpModal');
            if (helpModalElement) {
                const helpModal = new bootstrap.Modal(helpModalElement);
                helpModal.show();
            }
        });
    }

    // Add real-time validation for required fields
    if (titleInput) {
        titleInput.addEventListener('blur', function() {
            if (!this.value.trim()) {
                this.classList.add('is-invalid');
            } else {
                this.classList.remove('is-invalid');
            }
        });
    }

    if (descriptionInput) {
        descriptionInput.addEventListener('blur', function() {
            if (!this.value.trim()) {
                this.classList.add('is-invalid');
            } else {
                this.classList.remove('is-invalid');
            }
        });
    }

    // Add keyboard shortcut for preview (Ctrl/Cmd + P)
    document.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'p' && previewBtn) {
            e.preventDefault();
            previewBtn.click();
        }
    });

    // Auto-save draft functionality (optional)
    let autoSaveTimeout;
    function autoSaveDraft() {
        clearTimeout(autoSaveTimeout);
        autoSaveTimeout = setTimeout(() => {
            console.log('Auto-saving draft...');
            // Implement auto-save logic here
        }, 2000);
    }

    // Add auto-save listeners
    if (titleInput) titleInput.addEventListener('input', autoSaveDraft);
    if (descriptionInput) descriptionInput.addEventListener('input', autoSaveDraft);
    if (severitySelect) severitySelect.addEventListener('change', autoSaveDraft);
});