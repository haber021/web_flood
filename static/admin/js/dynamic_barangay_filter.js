(function($) {
    $(document).ready(function() {
        var $municipalitySelect = $('#id_municipality');
        var $barangaySelect = $('#id_barangay');

        function updateBarangayOptions() {
            var municipalityId = $municipalitySelect.val();
            if (municipalityId) {
                $.ajax({
                    url: '/api/all-barangays/?municipality_id=' + municipalityId,
                    type: 'GET',
                    success: function(data) {
                        $barangaySelect.empty();
                        $barangaySelect.append('<option value="">---------</option>');
                        $.each(data.barangays || [], function(index, barangay) {
                            $barangaySelect.append('<option value="' + barangay.id + '">' + barangay.name + '</option>');
                        });
                    },
                    error: function() {
                        console.error('Error fetching barangays');
                        $barangaySelect.empty();
                        $barangaySelect.append('<option value="">---------</option>');
                    }
                });
            } else {
                $barangaySelect.empty();
                $barangaySelect.append('<option value="">---------</option>');
            }
        }

        $municipalitySelect.change(updateBarangayOptions);

        // Initial load if municipality is already selected
        if ($municipalitySelect.val()) {
            updateBarangayOptions();
        }
    });
})(django.jQuery);