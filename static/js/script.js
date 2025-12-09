Document.addEventListener('DOMContentLoaded', () => {
    // --- Get DOM Elements ---
    const imageUpload = document.getElementById('imageUpload');
    const fileUploadLabel = document.querySelector('label[for="imageUpload"]');
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
    // Ensure the loading indicator element exists in HTML (as per previous step)
    const loadingIndicator = document.getElementById('imageLoadingIndicator');
    const resultImage = document.getElementById('resultImage');
    const downloadLink = document.getElementById('downloadLink');
    const tableResultArea = document.getElementById('table-result-area');
    const translationsTableBody = document.getElementById('translationsTable').querySelector('tbody');
    const processAnotherButton = document.getElementById('processAnotherButton');
    const modeExtractRadio = document.getElementById('modeExtract');
    const modeAutoRadio = document.getElementById('modeAuto');

    let selectedFile = null;
    let isConnected = false;

    // --- Initialize Socket.IO ---
    console.log("Initializing Socket.IO connection...");
    // Connects to the server that served the page
    const socket = io({
        transports: ['websocket', 'polling'], // Explicitly prefer WebSocket
        reconnectionAttempts: 5, // Attempt to reconnect 5 times
        reconnectionDelay: 2000, // Wait 2 seconds between attempts
    });

    // --- SocketIO Connection Event Listeners ---
    socket.on('connect', () => {
        isConnected = true;
        console.log('✅ Socket.IO connected! SID:', socket.id);
        if (selectedFile) {
            processButton.disabled = false; // Enable button if file already selected
            console.log("   Process button enabled (reconnected/file selected).");
        } else {
            console.log("   Waiting for file selection.");
            processButton.disabled = true; // Ensure button is disabled if no file
        }
        errorText.style.display = 'none'; // Hide connection errors if reconnected
    });

    socket.on('disconnect', (reason) => {
        isConnected = false;
        console.warn('❌ Socket.IO disconnected! Reason:', reason);
        processButton.disabled = true; // Disable button on disconnect
        // Display a non-alert message indicating disconnection
        if (reason !== 'io server disconnect') {
             errorText.textContent = "⚠️ تم قطع الاتصال بالخادم، جاري محاولة إعادة الاتصال...";
             errorText.style.display = 'block';
        }
        // Allow reconnection logic to handle UI reset if needed
    });

    socket.io.on('reconnect_attempt', (attempt) => {
        console.log(`   Socket.IO reconnect attempt ${attempt}...`);
        progressText.textContent = `⚠️ جاري محاولة إعادة الاتصال (${attempt})...`;
    });

    socket.io.on('reconnect_failed', () => {
        console.error('❌ Socket.IO reconnection failed!');
        alert("❌ فشلت إعادة الاتصال بالخادم. يرجى التحقق من اتصالك وتحديث الصفحة.");
        resetToUploadState(); // Reset fully if reconnection ultimately fails
    });


    socket.on('connect_error', (error) => {
         isConnected = false;
         console.error('❌ Socket.IO connection error:', error);
         // Display error without alert if possible
         errorText.textContent = "❌ فشل الاتصال بالخادم. تأكد أن الخادم يعمل.";
         errorText.style.display = 'block';
         processButton.disabled = true;
         resetToUploadState(); // Reset fully on initial connection error
    });

    // --- SocketIO Processing Status Listeners ---
    socket.on('processing_started', (data) => {
        console.log('Processing started:', data.message);
        // This might arrive quickly after upload, update text accordingly
        progressText.textContent = data.message || '⏳ بدأت المعالجة...';
        progressBar.value = 5; // Indicate processing has begun
    });

    socket.on('progress_update', (data) => {
        // Ensure percentage is valid
        const percentage = (data.percentage >= 0 && data.percentage <= 100) ? data.percentage : progressBar.value;
        progressBar.value = percentage;
        const stepPrefix = data.step >= 0 ? `[${data.step}/6] ` : ''; // Handle step -1 nicely
        progressText.textContent = `${stepPrefix}${data.message} (${percentage}%)`;
        errorText.style.display = 'none'; // Hide previous errors on progress
    });

    socket.on('processing_complete', (data) => {
        console.log('✅ Processing complete! Data:', data);
        progressBar.value = 100;
        progressText.textContent = '✨ اكتملت المعالجة! جارٍ تحميل صورة النتيجة...'; // Indicate image loading

        // Hide progress section after a short delay to show 100%
        setTimeout(() => {
             progressSection.style.display = 'none';
        }, 500); // 0.5 second delay

        // Prepare result section (show container, hide specific elements)
        resultSection.style.display = 'block';
        imageResultArea.style.display = 'none'; // Hide image container initially
        tableResultArea.style.display = 'none';
        translationsTableBody.innerHTML = '';
        resultImage.style.display = 'none'; // Hide img tag
        downloadLink.style.display = 'none'; // Hide download link

        // Show the image loading indicator (ensure it exists in HTML)
        if (loadingIndicator) loadingIndicator.style.display = 'block';

        // Validate received data
        if (!data || !data.mode || !data.imageUrl) {
            console.error("Invalid data received on completion", data);
            errorText.textContent = "خطأ: بيانات نتيجة غير صالحة من الخادم.";
            errorText.style.display = 'block';
            if (loadingIndicator) loadingIndicator.style.display = 'none'; // Hide spinner on error
            // Allow user to try again
            uploadSection.style.display = 'block';
            processButton.disabled = !(selectedFile && isConnected);
            resultSection.style.display = 'none'; // Hide incomplete result section
            return;
        }

        // Prepare display based on mode
        let baseDownloadName = generateDownloadFilename(selectedFile?.name, "");
        if (data.mode === 'extract') {
            console.log("   Preparing 'extract' results.");
            imageResultTitle.textContent = "الصورة المنظفة";
            downloadLink.download = baseDownloadName + "_cleaned.jpg";
            populateTable(data.translations); // Populate table
            tableResultArea.style.display = 'block'; // Show table
        } else if (data.mode === 'auto') {
            console.log("   Preparing 'auto' results.");
            imageResultTitle.textContent = "الصورة المترجمة تلقائياً";
             downloadLink.download = baseDownloadName + "_translated.jpg";
             // Table area remains hidden for 'auto' mode
        }
        imageResultArea.style.display = 'block'; // Show the container (spinner visible now)
        downloadLink.href = data.imageUrl; // Set download URL

        // --- Handle actual image loading ---
        resultImage.onload = () => {
            console.log("   Result image loaded successfully.");
            if (loadingIndicator) loadingIndicator.style.display = 'none'; // Hide spinner
            resultImage.style.display = 'block'; // Show the loaded image
            downloadLink.style.display = 'inline-block'; // Show download link
            progressText.textContent = '✨ اكتملت المعالجة!'; // Final status text
        };
        resultImage.onerror = (err) => {
            console.error("   Error loading result image from src:", data.imageUrl, err);
            if (loadingIndicator) loadingIndicator.style.display = 'none'; // Hide spinner
            // Display error within the image area
            const errorP = document.createElement('p');
            errorP.style.color = 'red';
            errorP.textContent = 'فشل تحميل صورة النتيجة.';
            imageResultArea.appendChild(errorP);

            downloadLink.style.display = 'none'; // Hide link if image fails
            progressText.textContent = '⚠️ اكتملت المعالجة ولكن فشل تحميل الصورة.';
        };

        // Set the src AFTER attaching onload/onerror to trigger loading
        console.log("   Setting result image src:", data.imageUrl);
        resultImage.src = data.imageUrl + '?t=' + Date.now(); // Cache bust

    });

    socket.on('processing_error', (data) => {
        console.error('❌ Processing Error:', data.error);
        errorText.textContent = `😭 خطأ في المعالجة: ${data.error}`;
        errorText.style.display = 'block';
        progressSection.style.display = 'none'; // Hide progress bar on error
        resultSection.style.display = 'none'; // Hide results
        uploadSection.style.display = 'block'; // Show upload section again
        // Re-enable button only if file still selected and connected
        processButton.disabled = !(selectedFile && isConnected);
    });

    // --- DOM Event Listeners ---
    imageUpload.addEventListener('change', (event) => {
        // Reset previous state when a new file is selected
        resetResultArea();
        errorText.style.display = 'none';

        selectedFile = event.target.files[0];
        console.log("File selected:", selectedFile);
        if (selectedFile) {
             const allowedTypes = ['image/png', 'image/jpeg', 'image/webp', 'application/zip'];
             const maxSizeMB = 9999999999999; // Match Flask config
             const maxSizeBytes = maxSizeMB * 1024 * 1024;

             // Validate Type
             if (!allowedTypes.includes(selectedFile.type)) {
                 alert(`نوع الملف غير صالح: ${selectedFile.type}.\nالأنواع المسموحة: PNG, JPG, WEBP.`);
                 resetFileSelection(); return;
             }
             // Validate Size
             if (selectedFile.size > maxSizeBytes) {
                 alert(`حجم الملف كبير جدًا (${(selectedFile.size / 1024 / 1024).toFixed(1)} MB).\nالحد الأقصى: ${maxSizeMB} MB.`);
                 resetFileSelection(); return;
             }

             fileNameSpan.textContent = selectedFile.name;
             processButton.disabled = !isConnected; // Enable button only if connected
             if (!isConnected) { console.warn("Socket not connected yet, process button disabled."); }

        } else {
            resetFileSelection(); // Clear selection if dialog cancelled
        }
    });

    // Allow clicking the label to trigger the hidden file input
    fileUploadLabel.addEventListener('click', (e) => {
        e.preventDefault();
        imageUpload.click();
    });

    // --- Process Button Click Handler (Using XMLHttpRequest) ---
    processButton.addEventListener('click', () => { // Does not need 'async'
        console.log("Process button clicked.");
        if (!selectedFile) { alert('الرجاء اختيار ملف صورة أولاً.'); return; }
        if (!isConnected) { alert('لا يوجد اتصال بالخادم. الرجاء الانتظار أو تحديث الصفحة.'); return; }

        const currentMode = modeAutoRadio.checked ? 'auto' : 'extract';
        console.log(`   Mode selected: ${currentMode}`);

        // --- Update UI for Uploading State ---
        uploadSection.style.display = 'none';
        progressSection.style.display = 'block';
        resultSection.style.display = 'none';
        errorText.style.display = 'none';
        progressBar.value = 0;
        progressText.textContent = '⏫ بدء الرفع... (0%)';
        processButton.disabled = true; // Disable button during upload/processing

        // --- Create FormData ---
        const formData = new FormData();
        formData.append('file', selectedFile); // Key 'file' MUST match backend

        // --- Use XMLHttpRequest for Upload Progress ---
        const xhr = new XMLHttpRequest();

        // --- Progress Event Listener ---
        xhr.upload.addEventListener('progress', (event) => {
            if (event.lengthComputable) {
                const percentage = Math.round((event.loaded / event.total) * 100);
                progressBar.value = percentage;
                progressText.textContent = `⏫ جارٍ رفع الصورة... (${percentage}%)`;
            } else {
                // Progress not computable
                progressText.textContent = '⏫ جارٍ رفع الصورة... (الحجم غير محدد)';
            }
        }, false); // Use capture=false (default)

        // --- Load Event Listener (Upload Complete/Server Responded) ---
        xhr.addEventListener('load', () => {
            console.log(`   XHR Upload finished with status: ${xhr.status}`);
            // Don't immediately set to 100% here, wait for server confirmation or processing start

            let resultJson;
            try {
                // Ensure response text exists before parsing
                if (!xhr.responseText) {
                     throw new Error("استجابة فارغة من الخادم.");
                }
                resultJson = JSON.parse(xhr.responseText);
            } catch (e) {
                 console.error("   ❌ Could not parse JSON response:", xhr.responseText, e);
                 errorText.textContent = `😭 خطأ: استجابة غير متوقعة من الخادم بعد الرفع. (${e.message})`;
                 errorText.style.display = 'block';
                 resetUiAfterError(true); // Reset UI allowing retry
                 return;
            }

            // Check HTTP status code for success (2xx)
            if (xhr.status >= 200 && xhr.status < 300) {
                // --- SUCCESS ---
                console.log("   ✅ Upload successful via XHR:", resultJson);
                progressBar.value = 100; // Visually complete upload bar
                progressText.textContent = '⏳ تم الرفع بنجاح، في انتظار بدء المعالجة...';

                const { output_filename_base, saved_filename } = resultJson;
                if (!output_filename_base || !saved_filename) {
                    console.error("   ❌ Incomplete data from server:", resultJson);
                    errorText.textContent = "😭 خطأ: بيانات ملف غير مكتملة من الخادم بعد الرفع.";
                    errorText.style.display = 'block';
                    resetUiAfterError(true);
                    return;
                }

                // Emit SocketIO event to start processing
                socket.emit('start_processing', {
                    output_filename_base: output_filename_base,
                    saved_filename: saved_filename,
                    mode: currentMode
                });
                console.log("   ✅ Emitted 'start_processing' via SocketIO.");
                // Keep button disabled, wait for socket 'processing_started' etc.

            } else {
                // --- ERROR from Server (e.g., 400, 500) ---
                console.error(`   ❌ Server returned error status ${xhr.status}:`, resultJson);
                errorText.textContent = `😭 خطأ الرفع: ${resultJson.error || ('خطأ من الخادم ' + xhr.status)}`;
                errorText.style.display = 'block';
                resetUiAfterError(true);
            }
        });

        // --- Error Event Listener (Network errors, CORS issues, etc.) ---
        xhr.addEventListener('error', (e) => {
            console.error("   ❌ XHR Upload failed (Network error or similar).", e);
            errorText.textContent = `😭 خطأ في الشبكة أو فشل في إرسال الملف. تأكد من اتصالك بالانترنت والخادم يعمل.`;
            errorText.style.display = 'block';
            resetUiAfterError(true);
        });

         // --- Abort Event Listener (Optional) ---
         xhr.addEventListener('abort', () => {
            console.warn("   XHR Upload aborted by user.");
            // Reset UI if upload is cancelled mid-way
            resetUiAfterError(true);
         });


        // --- Open and Send the Request ---
        try {
             console.log("   Opening and sending XHR POST request to /upload...");
             xhr.open('POST', '/upload', true); // true = asynchronous
             // Optional: Set a timeout for the upload request itself
             // xhr.timeout = 120000; // e.g., 120 seconds timeout
             // xhr.ontimeout = () => {
             //     console.error("   ❌ XHR Upload timed out.");
             //     errorText.textContent = `😭 خطأ: انتهت مهلة رفع الملف.`;
             //     errorText.style.display = 'block';
             //     resetUiAfterError(true);
             // };

             // Send the FormData
             xhr.send(formData);
        } catch (sendError) {
             // Catch synchronous errors during open/send (less common)
             console.error("   ❌ Error initiating XHR send:", sendError);
             errorText.textContent = `😭 خطأ غير متوقع عند محاولة رفع الملف.`;
             errorText.style.display = 'block';
             resetUiAfterError(true);
        }


    }); // End of processButton click listener

    // --- Process Another Button ---
    processAnotherButton.addEventListener('click', () => {
        console.log("Process Another clicked.");
        resetToUploadState();
    });

    // --- Helper Function to Reset UI after Upload/Processing Error ---
    // (Allows user to retry without re-selecting the file if possible)
    function resetUiAfterError(allowRetry = true) {
         progressSection.style.display = 'none'; // Hide progress
         uploadSection.style.display = 'block'; // Show upload controls again
         // Re-enable button only if a file is still selected and socket is connected
         if (allowRetry) {
              processButton.disabled = !(selectedFile && isConnected);
         } else {
              processButton.disabled = true;
         }
    }


    // --- Other Helper Functions ---
    function populateTable(translations) {
        translationsTableBody.innerHTML = '';
        if (!translations || translations.length === 0) {
            const row = translationsTableBody.insertRow();
            const cell = row.insertCell();
            cell.colSpan = 2;
            cell.textContent = "لم يتم استخراج أي نصوص أو ترجمات.";
            cell.style.textAlign = 'center';
            return;
        }
        translations.forEach(item => {
            const row = translationsTableBody.insertRow();
            const cellId = row.insertCell();
            const cellText = row.insertCell();
            cellId.textContent = (item.id !== undefined && item.id !== null) ? item.id : '-';
            const safeText = item.translation ? String(item.translation) : '';
            // Basic sanitization + newline handling
            cellText.innerHTML = safeText.replace(/</g, "&lt;")
                                        .replace(/>/g, "&gt;")
                                        .replace(/\n/g, '<br>');
        });
    }

    function generateDownloadFilename(originalName, suffix) {
        const defaultName = "processed_image";
        let baseName = defaultName;
        if (originalName && typeof originalName === 'string') {
            // Extract filename without extension, handle names with dots
            const lastDotIndex = originalName.lastIndexOf('.');
            if (lastDotIndex > 0) { // Ensure dot is not the first character
                 baseName = originalName.substring(0, lastDotIndex);
            } else if (lastDotIndex === -1) { // Handle names with no dots
                baseName = originalName;
            }
        }
        // Ensure suffix is added, default to jpg
        return `${baseName}${suffix || ''}.jpg`;
    }

    function resetFileSelection() {
        imageUpload.value = null;
        selectedFile = null;
        fileNameSpan.textContent = 'لم يتم اختيار أي ملف';
        processButton.disabled = true;
        console.log("File selection reset.");
    }

    function resetResultArea() {
        resultSection.style.display = 'none';
        imageResultArea.style.display = 'none';
        tableResultArea.style.display = 'none';
        resultImage.src = "#"; // Use '#' or '' to clear src
        resultImage.style.display = 'none'; // Ensure img tag is hidden
        downloadLink.href = "#";
        downloadLink.style.display = 'none'; // Ensure link is hidden
        // Clear any dynamically added error messages within image area
        const imgAreaError = imageResultArea.querySelector('p[style*="color: red;"]');
        if(imgAreaError) imgAreaError.remove();
        // Hide spinner if it was left visible
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        translationsTableBody.innerHTML = '';
        errorText.style.display = 'none'; // Hide main error text
        console.log("Result area reset.");
    }

    function resetToUploadState() {
        console.log("Resetting UI to initial upload state.");
        resetResultArea();
        resetFileSelection();
        progressSection.style.display = 'none';
        uploadSection.style.display = 'block';
        // Button state handled by connect/disconnect/file selection logic
        processButton.disabled = !(selectedFile && isConnected);
    }

    // --- Initial Page Load State ---
    resetToUploadState();
    console.log("Initial UI state set. Waiting for connection and file selection.");

}); // End DOMContentLoaded
