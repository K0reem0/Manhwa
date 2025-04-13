document.addEventListener('DOMContentLoaded', () => {
    // --- Get DOM Elements ---
    const imageUpload = document.getElementById('imageUpload');
    const fileNameSpan = document.getElementById('fileName');
    const processButton = document.getElementById('processButton');
    const uploadSection = document.getElementById('upload-section');
    const progressSection = document.getElementById('progress-section');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const errorText = document.getElementById('errorText');
    const resultSection = document.getElementById('result-section');
    const imageResultArea = document.getElementById('image-result-area');
    const imageResultTitle = document.getElementById('image-result-title');
    const resultImage = document.getElementById('resultImage');
    const downloadLink = document.getElementById('downloadLink');
    const tableResultArea = document.getElementById('table-result-area');
    const translationsTableBody = document.getElementById('translationsTable').querySelector('tbody');
    const processAnotherButton = document.getElementById('processAnotherButton');
    const modeExtractRadio = document.getElementById('modeExtract');
    const modeAutoRadio = document.getElementById('modeAuto');

    let selectedFile = null;
    let currentMode = 'extract'; // Default mode

    // --- Initialize Socket.IO ---
    const socket = io(); // Connects to the server that served the page

    // --- SocketIO Event Listeners ---
    socket.on('connect', () => {
        console.log('Connected to server via Socket.IO:', socket.id);
        // Enable button only if file is also selected
        if (selectedFile) {
            processButton.disabled = false;
        }
    });

    socket.on('disconnect', () => {
        console.warn('Disconnected from server');
        processButton.disabled = true; // Disable on disconnect
        alert("تم قطع الاتصال بالخادم. يرجى تحديث الصفحة.");
    });

    socket.on('processing_started', (data) => {
        console.log(data.message);
        progressText.textContent = data.message;
        progressBar.value = 5;
    });

    socket.on('progress_update', (data) => {
        // console.log('Progress:', data);
        progressBar.value = data.percentage;
        // Display step number if available, otherwise just message
        const stepPrefix = data.step ? `[${data.step}/6] ` : '';
        progressText.textContent = `${stepPrefix}${data.message} (${data.percentage}%)`;
        errorText.style.display = 'none'; // Hide error on progress
    });

    socket.on('processing_complete', (data) => {
        console.log('Processing complete:', data);
        progressText.textContent = 'اكتملت المعالجة!';
        progressBar.value = 100;

        // Hide progress, show results
        progressSection.style.display = 'none';
        resultSection.style.display = 'block';

        // Clear previous results
        imageResultArea.style.display = 'none';
        tableResultArea.style.display = 'none';
        translationsTableBody.innerHTML = ''; // Clear table

        if (data.mode === 'extract') {
            imageResultTitle.textContent = "الصورة المنظفة";
            resultImage.src = data.imageUrl + '?t=' + new Date().getTime(); // Cache bust
            downloadLink.href = data.imageUrl;
            downloadLink.download = generateDownloadFilename(selectedFile?.name, "_cleaned");
            imageResultArea.style.display = 'block';

            if (data.translations && data.translations.length > 0) {
                populateTable(data.translations);
                tableResultArea.style.display = 'block';
            } else {
                 // Optionally show a message if no translations were extracted
                 const row = translationsTableBody.insertRow();
                 const cell = row.insertCell();
                 cell.colSpan = 2; // Span across both columns
                 cell.textContent = "لم يتم استخراج أي نصوص.";
                 cell.style.textAlign = 'center';
                 tableResultArea.style.display = 'block';
            }

        } else if (data.mode === 'auto') {
            imageResultTitle.textContent = "الصورة المترجمة تلقائياً";
            resultImage.src = data.imageUrl + '?t=' + new Date().getTime(); // Cache bust
            downloadLink.href = data.imageUrl;
            downloadLink.download = generateDownloadFilename(selectedFile?.name, "_translated");
            imageResultArea.style.display = 'block';
            // Table remains hidden for auto mode
        }
    });

    socket.on('processing_error', (data) => {
        console.error('Processing Error:', data.error);
        errorText.textContent = `خطأ: ${data.error}`;
        errorText.style.display = 'block';
        // Keep progress section visible to show error, hide result
        progressSection.style.display = 'block';
        resultSection.style.display = 'none';
        // Re-enable upload section so user can retry
        uploadSection.style.display = 'block';
        processButton.disabled = false; // Re-enable button
    });

    // --- DOM Event Listeners ---
    imageUpload.addEventListener('change', (event) => {
        selectedFile = event.target.files[0];
        if (selectedFile) {
            fileNameSpan.textContent = selectedFile.name;
             // Enable button only if socket is connected
            processButton.disabled = !socket.connected;
             resetResultArea(); // Clear old results if a new file is chosen
        } else {
            fileNameSpan.textContent = 'لم يتم اختيار أي ملف';
            processButton.disabled = true;
        }
    });

     // Handle clicks on the hidden input's label
     fileNameSpan.previousElementSibling.addEventListener('click', () => {
          imageUpload.click();
     });


    processButton.addEventListener('click', () => {
        if (!selectedFile) {
            alert('الرجاء اختيار ملف صورة أولاً.');
            return;
        }

        // Determine selected mode
        currentMode = modeAutoRadio.checked ? 'auto' : 'extract';

        // Show progress, hide upload/result
        uploadSection.style.display = 'none';
        progressSection.style.display = 'block';
        resultSection.style.display = 'none';
        errorText.style.display = 'none';
        progressBar.value = 0;
        progressText.textContent = 'جارٍ رفع الصورة...';
        processButton.disabled = true; // Disable while processing

        // Read file as Base64 and send
        const reader = new FileReader();
        reader.onload = function(event) {
            const base64String = event.target.result;
            console.log(`Sending start_processing (Mode: ${currentMode})...`);
            socket.emit('start_processing', {
                file: base64String,
                mode: currentMode
            });
        };
        reader.onerror = function(error) {
             console.error("Error reading file:", error);
             alert("حدث خطأ أثناء قراءة الملف.");
             resetToUploadState();
        };
        reader.readAsDataURL(selectedFile);
    });

     processAnotherButton.addEventListener('click', () => {
         resetToUploadState();
     });

    // --- Helper Functions ---
    function populateTable(translations) {
        translationsTableBody.innerHTML = ''; // Clear previous entries
        translations.forEach(item => {
            const row = translationsTableBody.insertRow();
            const cellId = row.insertCell();
            const cellText = row.insertCell();
            cellId.textContent = item.id;
            // Preserve line breaks from translation if any
            cellText.innerHTML = item.translation.replace(/\n/g, '<br>');
        });
    }

     function generateDownloadFilename(originalName, suffix) {
         const defaultName = "processed_image";
         let baseName = defaultName;
         if (originalName) {
             // Remove extension from original name
             baseName = originalName.split('.').slice(0, -1).join('.');
             if (!baseName) baseName = defaultName; // Handle names like ".png"
         }
         return `${baseName}${suffix}.jpg`; // Assume JPG output
     }

    function resetResultArea() {
        resultSection.style.display = 'none';
        imageResultArea.style.display = 'none';
        tableResultArea.style.display = 'none';
        resultImage.src = "#";
        downloadLink.href = "#";
        translationsTableBody.innerHTML = '';
    }

     function resetToUploadState() {
         resetResultArea();
         progressSection.style.display = 'none';
         uploadSection.style.display = 'block';
         imageUpload.value = null; // Clear file input visually
         selectedFile = null;
         fileNameSpan.textContent = 'لم يتم اختيار أي ملف';
         processButton.disabled = true; // Disable until file selected again
         errorText.style.display = 'none';
     }

});

