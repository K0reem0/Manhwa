document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const translateCheckbox = document.getElementById('translateCheckbox');
    const processButton = document.getElementById('processButton');
    const uploadSection = document.getElementById('upload-section');
    const progressSection = document.getElementById('progress-section');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const errorText = document.getElementById('errorText');
    const resultSection = document.getElementById('result-section');
    const resultImage = document.getElementById('resultImage');
    const downloadLink = document.getElementById('downloadLink');

    let selectedFile = null;

    // Initialize Socket.IO connection
    // Use ws:// or wss:// depending on your deployment (http vs https)
    const socket = io(); // Connects to the same server that served the page

    socket.on('connect', () => {
        console.log('Connected to server via Socket.IO');
        // Enable upload button once connected and file is selected
        if (selectedFile) {
            processButton.disabled = false;
        }
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        // Optionally disable button or show message
        processButton.disabled = true;
        alert("Connection lost. Please refresh the page.");
    });

    imageUpload.addEventListener('change', (event) => {
        selectedFile = event.target.files[0];
        if (selectedFile && socket.connected) {
            processButton.disabled = false;
        } else {
            processButton.disabled = true;
        }
        // Reset UI if a new file is chosen
        resetUI();
    });

    processButton.addEventListener('click', () => {
        if (!selectedFile) {
            alert('Please select an image file first.');
            return;
        }

        // Show progress, hide upload/result
        uploadSection.style.display = 'none';
        progressSection.style.display = 'block';
        resultSection.style.display = 'none';
        errorText.style.display = 'none';
        progressBar.value = 0;
        progressText.textContent = 'Uploading...';
        processButton.disabled = true; // Disable while processing

        // Read file as Base64
        const reader = new FileReader();
        reader.onload = function(event) {
            const base64String = event.target.result;
            const doTranslate = translateCheckbox.checked;

            // Send data via Socket.IO
            console.log('Sending start_processing event...');
            socket.emit('start_processing', {
                file: base64String, // Send base64 data
                translate: doTranslate
            });
        };
        reader.onerror = function(error) {
             console.error("Error reading file:", error);
             alert("Error reading file.");
             resetUI(); // Go back to upload state
        };
        reader.readAsDataURL(selectedFile); // Read as Data URL (includes base64)
    });

    // Listen for progress updates from server
    socket.on('progress_update', (data) => {
        console.log('Progress:', data);
        progressBar.value = data.percentage;
        progressText.textContent = `Step ${data.step}: ${data.message} (${data.percentage}%)`;
        errorText.style.display = 'none'; // Hide error if progress occurs
    });

    // Listen for completion event
    socket.on('processing_complete', (data) => {
        console.log('Processing complete:', data);
        progressText.textContent = 'Done!';
        progressBar.value = 100;
        resultImage.src = data.result_url + '?t=' + new Date().getTime(); // Add timestamp to prevent caching
        downloadLink.href = data.result_url;
        downloadLink.download = selectedFile.name.replace(/\.[^/.]+$/, "") + "_processed.jpg"; // Suggest a download name

        progressSection.style.display = 'none';
        resultSection.style.display = 'block';
        // Re-enable button for next file AFTER result is shown
         // processButton.disabled = false; // Keep disabled until new file selected
        imageUpload.value = null; // Clear the file input
        selectedFile = null; // Reset selected file
         uploadSection.style.display = 'block'; // Show upload again
    });

    // Listen for error events
    socket.on('processing_error', (data) => {
        console.error('Processing Error:', data.error);
        errorText.textContent = `Error: ${data.error}`;
        errorText.style.display = 'block';
        progressSection.style.display = 'block'; // Keep progress section visible to show error
        resultSection.style.display = 'none';
        // Re-enable button and show upload section so user can retry
        processButton.disabled = false;
        uploadSection.style.display = 'block';

    });

     socket.on('processing_started', (data) => {
         console.log(data.message);
         progressText.textContent = data.message;
         progressBar.value = 5; // Indicate it has started
     });

    function resetUI() {
        progressSection.style.display = 'none';
        resultSection.style.display = 'none';
        uploadSection.style.display = 'block';
        errorText.style.display = 'none';
        progressBar.value = 0;
        progressText.textContent = 'Starting...';
        resultImage.src = "#"; // Clear previous image
        downloadLink.href = "#";
         // Enable button only if file selected AND socket connected
        processButton.disabled = !(selectedFile && socket.connected);
    }

});
