document.addEventListener('DOMContentLoaded', () => {
    // --- Get DOM Elements ---
    const imageUpload = document.getElementById('imageUpload');
    const fileUploadLabel = document.querySelector('label[for="imageUpload"]'); // Get the label acting as button
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
    let isConnected = false; // Track connection status

    // --- Initialize Socket.IO ---
    console.log("Initializing Socket.IO connection...");
    const socket = io(); // Connects to the server that served the page

    // --- SocketIO Event Listeners ---
    socket.on('connect', () => {
        isConnected = true;
        console.log('✅ Socket.IO connected! SID:', socket.id);
        // Enable button only if file is also selected
        if (selectedFile) {
            processButton.disabled = false;
            console.log("   Process button enabled (file selected + connected).");
        } else {
            console.log("   Socket connected, waiting for file selection.");
        }
    });

    socket.on('disconnect', (reason) => {
        isConnected = false;
        console.warn('❌ Socket.IO disconnected! Reason:', reason);
        processButton.disabled = true; // Disable on disconnect
        alert("⚠️ تم قطع الاتصال بالخادم. يرجى تحديث الصفحة. Reason: " + reason);
        resetToUploadState(); // Reset UI on disconnect
    });

    socket.on('connect_error', (error) => {
         isConnected = false;
         console.error('❌ Socket.IO connection error:', error);
         alert("❌ فشل الاتصال بالخادم. يرجى التأكد من أن الخادم يعمل وتحديث الصفحة.");
         processButton.disabled = true;
         resetToUploadState();
    });

    // -- Processing Handlers --
    socket.on('processing_started', (data) => {
        console.log('Processing started message received:', data.message);
        progressText.textContent = data.message; // Update text based on server message
        progressBar.value = 5; // Show minimal progress
    });

    socket.on('progress_update', (data) => {
        // console.log('Progress:', data); // Reduce console noise
        progressBar.value = data.percentage;
        const stepPrefix = data.step ? `[${data.step}/6] ` : '';
        progressText.textContent = `${stepPrefix}${data.message} (${data.percentage}%)`;
        errorText.style.display = 'none';
    });

    socket.on('processing_complete', (data) => {
        console.log('✅ Processing complete! Data received:', data);
        progressText.textContent = '✨ اكتملت المعالجة!';
        progressBar.value = 100;

        progressSection.style.display = 'none';
        resultSection.style.display = 'block';

        imageResultArea.style.display = 'none';
        tableResultArea.style.display = 'none';
        translationsTableBody.innerHTML = '';

        if (!data || !data.mode || !data.imageUrl) {
            console.error("Error: Invalid data received on processing_complete", data);
            errorText.textContent = "خطأ: بيانات نتيجة غير صالحة من الخادم.";
            errorText.style.display = 'block';
            return;
        }

        // Display results based on mode
        if (data.mode === 'extract') {
            console.log("   Displaying results for 'extract' mode.");
            imageResultTitle.textContent = "الصورة المنظفة";
            resultImage.src = data.imageUrl + '?t=' + new Date().getTime();
            downloadLink.href = data.imageUrl;
            downloadLink.download = generateDownloadFilename(selectedFile?.name, "_cleaned");
            imageResultArea.style.display = 'block';

            if (data.translations && data.translations.length > 0) {
                populateTable(data.translations);
                tableResultArea.style.display = 'block';
            } else {
                 const row = translationsTableBody.insertRow();
                 const cell = row.insertCell();
                 cell.colSpan = 2;
                 cell.textContent = "لم يتم استخراج أي نصوص.";
                 cell.style.textAlign = 'center';
                 tableResultArea.style.display = 'block';
            }

        } else if (data.mode === 'auto') {
            console.log("   Displaying results for 'auto' mode.");
            imageResultTitle.textContent = "الصورة المترجمة تلقائياً";
            resultImage.src = data.imageUrl + '?t=' + new Date().getTime();
            downloadLink.href = data.imageUrl;
            downloadLink.download = generateDownloadFilename(selectedFile?.name, "_translated");
            imageResultArea.style.display = 'block';
        }
    });

    socket.on('processing_error', (data) => {
        console.error('❌ Processing Error Received:', data.error);
        errorText.textContent = `😭 خطأ: ${data.error}`;
        errorText.style.display = 'block';
        progressSection.style.display = 'block'; // Keep progress section visible
        progressBar.value = 0; // Reset progress bar
        resultSection.style.display = 'none';
        uploadSection.style.display = 'block'; // Show upload section to allow retry
        processButton.disabled = false; // Re-enable button after error
    });

    // --- DOM Event Listeners ---
    imageUpload.addEventListener('change', (event) => {
        selectedFile = event.target.files[0];
        console.log("File selected:", selectedFile);
        if (selectedFile) {
            // Basic file type check (though server validates again)
             const allowedTypes = ['image/png', 'image/jpeg', 'image/webp'];
             if (!allowedTypes.includes(selectedFile.type)) {
                 alert(`نوع الملف غير صالح: ${selectedFile.type}. الأنواع المسموح بها: PNG, JPG, WEBP`);
                 resetFileSelection();
                 return;
             }
             // Basic size check (e.g., 16MB)
             if (selectedFile.size > 16 * 1024 * 1024) {
                  alert(`حجم الملف كبير جدًا (${(selectedFile.size / 1024 / 1024).toFixed(1)} MB). الحد الأقصى 16MB.`);
                  resetFileSelection();
                  return;
             }

            fileNameSpan.textContent = selectedFile.name;
            processButton.disabled = !isConnected; // Enable only if connected
            if (!isConnected) {
                console.warn("File selected, but socket not connected yet.");
            }
            resetResultArea();
        } else {
            resetFileSelection();
        }
    });

     // Trigger hidden file input when the label button is clicked
     fileUploadLabel.addEventListener('click', (e) => {
          e.preventDefault(); // Prevent default label behavior if any
          imageUpload.click();
     });

    processButton.addEventListener('click', () => {
        console.log("Process button clicked.");
        if (!selectedFile) {
            alert('الرجاء اختيار ملف صورة أولاً.');
            console.log("   Aborted: No file selected.");
            return;
        }
        if (!isConnected) {
             alert('لا يوجد اتصال بالخادم. يرجى الانتظار أو تحديث الصفحة.');
             console.log("   Aborted: Socket not connected.");
             return;
        }

        const currentMode = modeAutoRadio.checked ? 'auto' : 'extract';
        console.log(`   Mode selected: ${currentMode}`);

        // --- Update UI ---
        uploadSection.style.display = 'none';
        progressSection.style.display = 'block';
        resultSection.style.display = 'none';
        errorText.style.display = 'none';
        progressBar.value = 0;
        progressText.textContent = '⏳ جارٍ قراءة الملف...'; // More specific initial text
        processButton.disabled = true;

        // --- Read File and Emit ---
        const reader = new FileReader();
        reader.onload = function(event) {
            try {
                const base64String = event.target.result;
                console.log("   FileReader loaded. Read as Data URL.");
                // console.log("   Base64 Data Length:", base64String.length); // Optional: log length

                if (!base64String || !base64String.startsWith('data:image')) {
                     throw new Error("Invalid Data URL generated.");
                }

                console.log(`   Attempting to emit 'start_processing' (Mode: ${currentMode})...`);
                progressText.textContent = '⏫ جارٍ رفع الصورة...'; // Update text before emit

                socket.emit('start_processing', {
                    file: base64String,
                    mode: currentMode
                });
                console.log("   ✅ 'start_processing' event emitted.");

            } catch (error) {
                 console.error("   ❌ Error during file processing or emit:", error);
                 alert("حدث خطأ أثناء تجهيز الملف للإرسال: " + error.message);
                 resetToUploadState(); // Reset UI on error
            }
        };
        reader.onerror = function(error) {
             console.error("   ❌ FileReader error:", error);
             alert("حدث خطأ أثناء قراءة الملف: " + error.message);
             resetToUploadState();
        };

        // Start reading the file
        console.log("   Calling reader.readAsDataURL()...");
        reader.readAsDataURL(selectedFile);
    });

     processAnotherButton.addEventListener('click', () => {
         console.log("Process Another button clicked.");
         resetToUploadState();
     });

    // --- Helper Functions ---
    function populateTable(translations) {
        translationsTableBody.innerHTML = '';
        if (!translations || translations.length === 0) {
            // Handle empty translations case visually in the table
             const row = translationsTableBody.insertRow();
             const cell = row.insertCell();
             cell.colSpan = 2;
             cell.textContent = "لم يتم استخراج أي نصوص.";
             cell.style.textAlign = 'center';
             return;
        }
        translations.forEach(item => {
            const row = translationsTableBody.insertRow();
            const cellId = row.insertCell();
            const cellText = row.insertCell();
            cellId.textContent = item.id !== undefined ? item.id : '-'; // Handle missing ID
            // Sanitize and display text, preserving line breaks safely
            const safeText = item.translation ? String(item.translation) : '';
            cellText.innerHTML = safeText.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, '<br>');
        });
    }

     function generateDownloadFilename(originalName, suffix) {
         const defaultName = "processed_image";
         let baseName = defaultName;
         if (originalName && typeof originalName === 'string') {
             baseName = originalName.split('.').slice(0, -1).join('.') || defaultName;
         }
         return `${baseName}${suffix}.jpg`;
     }

     function resetFileSelection() {
         imageUpload.value = null; // Clear the actual input
         selectedFile = null;
         fileNameSpan.textContent = 'لم يتم اختيار أي ملف';
         processButton.disabled = true; // Disable button
         console.log("File selection reset.");
     }

    function resetResultArea() {
        resultSection.style.display = 'none';
        imageResultArea.style.display = 'none';
        tableResultArea.style.display = 'none';
        resultImage.src = "#";
        downloadLink.href = "#";
        translationsTableBody.innerHTML = '';
        errorText.style.display = 'none'; // Hide errors when resetting
        console.log("Result area reset.");
    }

     function resetToUploadState() {
         console.log("Resetting UI to upload state.");
         resetResultArea();
         resetFileSelection();
         progressSection.style.display = 'none';
         uploadSection.style.display = 'block';
         // Button should be disabled by resetFileSelection
     }

     // Initial UI state setup
     resetToUploadState();
     console.log("Initial UI state set.");

}); // End DOMContentLoaded
