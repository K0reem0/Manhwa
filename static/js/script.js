document.addEventListener('DOMContentLoaded', () => {
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
    const loadingIndicator = document.getElementById('imageLoadingIndicator');
    const resultImage = document.getElementById('resultImage');
    const downloadLink = document.getElementById('downloadLink');
    const tableResultArea = document.getElementById('table-result-area');
    const translationsTableBody = document.getElementById('translationsTable').querySelector('tbody');
    const processAnotherButton = document.getElementById('processAnotherButton');
    const modeExtractRadio = document.getElementById('modeExtract');
    const modeAutoRadio = document.getElementById('modeAuto');

    // --- NEW ELEMENTS FOR BATCH PROCESSING ---
    const batchSummarySection = document.getElementById('batch-summary-section');
    const batchSummaryText = document.getElementById('batchSummaryText');
    const batchResultsContainer = document.getElementById('batch-results-container');
    const batchProgressBar = document.getElementById('batchProgressBar');

    let selectedFile = null;
    let isConnected = false;
    // NEW: Variable to hold image list for batch processing
    let imagesForBatch = [];
    let processedImageCount = 0;
    let totalImagesInBatch = 0;

    // --- Initialize Socket.IO ---
    console.log("Initializing Socket.IO connection...");
    const socket = io({
        transports: ['websocket', 'polling'],
        reconnectionAttempts: 5,
        reconnectionDelay: 2000,
    });

    // --- SocketIO Connection Event Listeners ---
    socket.on('connect', () => {
        isConnected = true;
        console.log('✅ Socket.IO connected! SID:', socket.id);
        if (selectedFile) {
            processButton.disabled = false;
        } else {
            processButton.disabled = true;
        }
        errorText.style.display = 'none';
    });

    socket.on('disconnect', (reason) => {
        isConnected = false;
        console.warn('❌ Socket.IO disconnected! Reason:', reason);
        processButton.disabled = true;
        if (reason !== 'io server disconnect') {
             errorText.textContent = "⚠️ تم قطع الاتصال بالخادم، جاري محاولة إعادة الاتصال...";
             errorText.style.display = 'block';
        }
    });

    socket.io.on('reconnect_attempt', (attempt) => {
        console.log(`   Socket.IO reconnect attempt ${attempt}...`);
        progressText.textContent = `⚠️ جاري محاولة إعادة الاتصال (${attempt})...`;
    });

    socket.io.on('reconnect_failed', () => {
        console.error('❌ Socket.IO reconnection failed!');
        alert("❌ فشلت إعادة الاتصال بالخادم. يرجى التحقق من اتصالك وتحديث الصفحة.");
        resetToUploadState();
    });

    socket.on('connect_error', (error) => {
         isConnected = false;
         console.error('❌ Socket.IO connection error:', error);
         errorText.textContent = "❌ فشل الاتصال بالخادم. تأكد أن الخادم يعمل.";
         errorText.style.display = 'block';
         processButton.disabled = true;
         resetToUploadState();
    });

    // --- SocketIO Processing Status Listeners ---
    socket.on('processing_started', (data) => {
        console.log('Processing started:', data.message);
        progressText.textContent = data.message || '⏳ بدأت المعالجة...';
        progressBar.value = 5;
    });

    socket.on('progress_update', (data) => {
        const percentage = (data.percentage >= 0 && data.percentage <= 100) ? data.percentage : progressBar.value;
        progressBar.value = percentage;
        const stepPrefix = data.step >= 0 ? `[${data.step}/6] ` : '';
        progressText.textContent = `${stepPrefix}${data.message} (${percentage}%)`;
        errorText.style.display = 'none';
    });
    
    // --- MODIFIED: Handle single image completion for both single and batch tasks ---
    socket.on('processing_complete', (data) => {
        console.log('✅ Processing complete for a single image! Data:', data);
        processedImageCount++;
        
        // Check if this is part of a batch
        if (data.is_zip_batch) {
             console.log(`   Batch Image Complete: ${data.original_filename}. Processed ${processedImageCount}/${totalImagesInBatch}`);
             updateBatchProgress();
             displayBatchResult(data);
             if (processedImageCount === totalImagesInBatch) {
                 console.log("   All images in batch are complete!");
                 batchSummaryText.textContent = `✨ اكتملت معالجة الدفعة! تمت معالجة ${totalImagesInBatch} صورة.`;
                 // Show 'process another' button for a new task
                 processAnotherButton.style.display = 'inline-block';
             }
        } else { // Single image mode
            progressBar.value = 100;
            progressText.textContent = '✨ اكتملت المعالجة! جارٍ تحميل صورة النتيجة...';
            setTimeout(() => progressSection.style.display = 'none', 500);

            resultSection.style.display = 'block';
            imageResultArea.style.display = 'none';
            tableResultArea.style.display = 'none';
            translationsTableBody.innerHTML = '';
            resultImage.style.display = 'none';
            downloadLink.style.display = 'none';
            if (loadingIndicator) loadingIndicator.style.display = 'block';

            if (!data || !data.mode || !data.imageUrl) {
                console.error("Invalid data received on completion", data);
                errorText.textContent = "خطأ: بيانات نتيجة غير صالحة من الخادم.";
                errorText.style.display = 'block';
                if (loadingIndicator) loadingIndicator.style.display = 'none';
                resetUiAfterError(true);
                return;
            }

            let baseDownloadName = generateDownloadFilename(selectedFile?.name, "");
            if (data.mode === 'extract') {
                imageResultTitle.textContent = "الصورة المنظفة";
                downloadLink.download = baseDownloadName + "_cleaned.jpg";
                populateTable(data.translations);
                tableResultArea.style.display = 'block';
            } else if (data.mode === 'auto') {
                imageResultTitle.textContent = "الصورة المترجمة تلقائياً";
                downloadLink.download = baseDownloadName + "_translated.jpg";
            }
            imageResultArea.style.display = 'block';
            downloadLink.href = data.imageUrl;

            resultImage.onload = () => {
                console.log("   Result image loaded successfully.");
                if (loadingIndicator) loadingIndicator.style.display = 'none';
                resultImage.style.display = 'block';
                downloadLink.style.display = 'inline-block';
                progressText.textContent = '✨ اكتملت المعالجة!';
            };
            resultImage.onerror = (err) => {
                console.error("   Error loading result image from src:", data.imageUrl, err);
                if (loadingIndicator) loadingIndicator.style.display = 'none';
                const errorP = document.createElement('p');
                errorP.style.color = 'red';
                errorP.textContent = 'فشل تحميل صورة النتيجة.';
                imageResultArea.appendChild(errorP);
                downloadLink.style.display = 'none';
                progressText.textContent = '⚠️ اكتملت المعالجة ولكن فشل تحميل الصورة.';
            };
            resultImage.src = data.imageUrl + '?t=' + Date.now();
        }
    });

    // --- NEW: Handle batch processing start ---
    socket.on('batch_started', (data) => {
         console.log('Batch processing started:', data);
         totalImagesInBatch = data.total_images;
         processedImageCount = 0;
         batchProgressBar.value = 0;
         batchSummaryText.textContent = `جاري معالجة ${totalImagesInBatch} صورة...`;

         // Hide single result areas and show batch area
         uploadSection.style.display = 'none';
         progressSection.style.display = 'none';
         resultSection.style.display = 'none';
         batchSummarySection.style.display = 'block';
         batchResultsContainer.innerHTML = ''; // Clear previous results
         processAnotherButton.style.display = 'none'; // Hide until batch complete
    });
    
    socket.on('processing_error', (data) => {
        console.error('❌ Processing Error:', data.error);
        errorText.textContent = `😭 خطأ في المعالجة: ${data.error}`;
        errorText.style.display = 'block';
        
        // Check if this is a single file or part of a batch
        if (totalImagesInBatch > 0) {
            // In a batch, just log the error and continue
            processedImageCount++;
            updateBatchProgress();
            const errorDiv = document.createElement('div');
            errorDiv.className = 'batch-result-item error';
            errorDiv.innerHTML = `<p class="filename">⚠️ خطأ في معالجة الملف.</p><p>${data.error}</p>`;
            batchResultsContainer.appendChild(errorDiv);
        } else {
            // For a single file, reset the UI
            progressSection.style.display = 'none';
            resultSection.style.display = 'none';
            uploadSection.style.display = 'block';
            processButton.disabled = !(selectedFile && isConnected);
        }
    });

    // --- DOM Event Listeners ---
    imageUpload.addEventListener('change', (event) => {
        resetResultArea();
        errorText.style.display = 'none';

        selectedFile = event.target.files[0];
        console.log("File selected:", selectedFile);

        if (selectedFile) {
             // MODIFIED: Added application/zip to allowed types
             const allowedTypes = ['image/png', 'image/jpeg', 'image/webp', 'application/zip'];
             const maxSizeMB = 50; // New limit for zip files
             const maxSizeBytes = maxSizeMB * 1024 * 1024;
             
             // Validate Type
             if (!allowedTypes.includes(selectedFile.type)) {
                 alert(`نوع الملف غير صالح: ${selectedFile.type}.\nالأنواع المسموحة: PNG, JPG, WEBP, ZIP.`);
                 resetFileSelection(); return;
             }
             // Validate Size
             if (selectedFile.size > maxSizeBytes) {
                 alert(`حجم الملف كبير جدًا (${(selectedFile.size / 1024 / 1024).toFixed(1)} MB).\nالحد الأقصى: ${maxSizeMB} MB.`);
                 resetFileSelection(); return;
             }
             
             fileNameSpan.textContent = selectedFile.name;
             processButton.disabled = !isConnected;
             if (!isConnected) { console.warn("Socket not connected yet, process button disabled."); }
        } else {
            resetFileSelection();
        }
    });

    fileUploadLabel.addEventListener('click', (e) => {
        e.preventDefault();
        imageUpload.click();
    });

    processButton.addEventListener('click', () => {
        console.log("Process button clicked.");
        if (!selectedFile) { alert('الرجاء اختيار ملف أولاً.'); return; }
        if (!isConnected) { alert('لا يوجد اتصال بالخادم. الرجاء الانتظار أو تحديث الصفحة.'); return; }
        
        // MODIFIED: Determine upload endpoint based on file type
        let uploadUrl = '/upload';
        if (selectedFile.type === 'application/zip') {
             uploadUrl = '/upload_zip';
        }
        console.log(`   Selected file type: ${selectedFile.type}. Using upload URL: ${uploadUrl}`);

        const currentMode = modeAutoRadio.checked ? 'auto' : 'extract';
        console.log(`   Mode selected: ${currentMode}`);

        uploadSection.style.display = 'none';
        progressSection.style.display = 'block';
        resultSection.style.display = 'none';
        batchSummarySection.style.display = 'none'; // Hide batch summary
        errorText.style.display = 'none';
        progressBar.value = 0;
        progressText.textContent = '⏫ بدء الرفع... (0%)';
        processButton.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);

        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener('progress', (event) => {
            if (event.lengthComputable) {
                const percentage = Math.round((event.loaded / event.total) * 100);
                progressBar.value = percentage;
                progressText.textContent = `⏫ جارٍ رفع الملف... (${percentage}%)`;
            } else {
                progressText.textContent = '⏫ جارٍ رفع الملف... (الحجم غير محدد)';
            }
        }, false);

        xhr.addEventListener('load', () => {
            console.log(`   XHR Upload finished with status: ${xhr.status}`);
            let resultJson;
            try {
                if (!xhr.responseText) throw new Error("استجابة فارغة من الخادم.");
                resultJson = JSON.parse(xhr.responseText);
            } catch (e) {
                 console.error("   ❌ Could not parse JSON response:", xhr.responseText, e);
                 errorText.textContent = `😭 خطأ: استجابة غير متوقعة من الخادم بعد الرفع. (${e.message})`;
                 errorText.style.display = 'block';
                 resetUiAfterError(true);
                 return;
            }

            if (xhr.status >= 200 && xhr.status < 300) {
                console.log("   ✅ Upload successful via XHR:", resultJson);
                progressBar.value = 100;
                progressText.textContent = '⏳ تم الرفع بنجاح، في انتظار بدء المعالجة...';

                // MODIFIED: Check if response contains 'images_to_process' (batch) or 'output_filename_base' (single)
                if (resultJson.images_to_process) {
                    console.log("   Batch upload detected. Emitting 'start_batch_processing'.");
                    imagesForBatch = resultJson.images_to_process;
                    socket.emit('start_batch_processing', {
                        images_to_process: imagesForBatch,
                        mode: currentMode
                    });
                } else if (resultJson.output_filename_base) {
                    console.log("   Single image upload detected. Emitting 'start_processing'.");
                    socket.emit('start_processing', {
                        output_filename_base: resultJson.output_filename_base,
                        saved_filename: resultJson.saved_filename,
                        mode: currentMode
                    });
                } else {
                    console.error("   ❌ Server response missing required data for both single and batch.");
                    errorText.textContent = "😭 خطأ: بيانات ملف غير مكتملة من الخادم بعد الرفع.";
                    errorText.style.display = 'block';
                    resetUiAfterError(true);
                }
            } else {
                console.error(`   ❌ Server returned error status ${xhr.status}:`, resultJson);
                errorText.textContent = `😭 خطأ الرفع: ${resultJson.error || ('خطأ من الخادم ' + xhr.status)}`;
                errorText.style.display = 'block';
                resetUiAfterError(true);
            }
        });

        xhr.addEventListener('error', (e) => {
            console.error("   ❌ XHR Upload failed (Network error or similar).", e);
            errorText.textContent = `😭 خطأ في الشبكة أو فشل في إرسال الملف. تأكد من اتصالك بالانترنت والخادم يعمل.`;
            errorText.style.display = 'block';
            resetUiAfterError(true);
        });

        try {
             xhr.open('POST', uploadUrl, true);
             xhr.send(formData);
        } catch (sendError) {
             console.error("   ❌ Error initiating XHR send:", sendError);
             errorText.textContent = `😭 خطأ غير متوقع عند محاولة رفع الملف.`;
             errorText.style.display = 'block';
             resetUiAfterError(true);
        }
    });

    processAnotherButton.addEventListener('click', () => {
        console.log("Process Another clicked.");
        resetToUploadState();
    });
    
    // --- NEW: Helper function to update batch progress bar ---
    function updateBatchProgress() {
        const percentage = Math.round((processedImageCount / totalImagesInBatch) * 100);
        batchProgressBar.value = percentage;
        batchSummaryText.textContent = `جاري معالجة ${processedImageCount}/${totalImagesInBatch} صورة (${percentage}%)`;
    }

    // --- NEW: Helper function to display results for a single image in a batch ---
    function displayBatchResult(data) {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'batch-result-item';

        const title = document.createElement('h3');
        title.className = 'filename';
        title.textContent = data.original_filename;
        itemDiv.appendChild(title);
        
        const modeText = document.createElement('p');
        modeText.className = 'mode-info';
        modeText.textContent = data.mode === 'auto' ? 'وضع تلقائي (ترجمة)' : 'وضع استخراج (تنظيف)';
        itemDiv.appendChild(modeText);

        if (data.mode === 'auto') {
            const resultImage = document.createElement('img');
            resultImage.src = data.imageUrl;
            resultImage.alt = 'Translated Image';
            itemDiv.appendChild(resultImage);
        } else if (data.mode === 'extract') {
            const downloadLink = document.createElement('a');
            downloadLink.href = data.imageUrl;
            downloadLink.download = generateDownloadFilename(data.original_filename, "_cleaned");
            downloadLink.textContent = 'تنزيل الصورة المنظفة';
            downloadLink.className = 'download-link';
            itemDiv.appendChild(downloadLink);
            
            const translationsList = document.createElement('ul');
            if (data.translations && data.translations.length > 0) {
                 data.translations.forEach(t => {
                     const li = document.createElement('li');
                     li.innerHTML = `<span class="translation-id">#${t.id}:</span> ${t.translation}`;
                     translationsList.appendChild(li);
                 });
                 itemDiv.appendChild(translationsList);
            } else {
                 const p = document.createElement('p');
                 p.textContent = 'لم يتم العثور على ترجمات.';
                 itemDiv.appendChild(p);
            }
        }
        
        batchResultsContainer.appendChild(itemDiv);
    }
    
    // --- Helper Function to Reset UI after Upload/Processing Error ---
    function resetUiAfterError(allowRetry = true) {
         progressSection.style.display = 'none';
         uploadSection.style.display = 'block';
         batchSummarySection.style.display = 'none'; // Hide batch area
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
            cellText.innerHTML = safeText.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, '<br>');
        });
    }

    function generateDownloadFilename(originalName, suffix) {
        const defaultName = "processed_image";
        let baseName = defaultName;
        if (originalName && typeof originalName === 'string') {
            const lastDotIndex = originalName.lastIndexOf('.');
            if (lastDotIndex > 0) {
                 baseName = originalName.substring(0, lastDotIndex);
            } else if (lastDotIndex === -1) {
                baseName = originalName;
            }
        }
        return `${baseName}${suffix || ''}.jpg`;
    }

    function resetFileSelection() {
        imageUpload.value = null;
        selectedFile = null;
        fileNameSpan.textContent = 'لم يتم اختيار أي ملف';
        processButton.disabled = true;
        imagesForBatch = [];
        console.log("File selection reset.");
    }

    function resetResultArea() {
        resultSection.style.display = 'none';
        imageResultArea.style.display = 'none';
        tableResultArea.style.display = 'none';
        resultImage.src = "#";
        resultImage.style.display = 'none';
        downloadLink.href = "#";
        downloadLink.style.display = 'none';
        const imgAreaError = imageResultArea.querySelector('p[style*="color: red;"]');
        if(imgAreaError) imgAreaError.remove();
        if (loadingIndicator) loadingIndicator.style.display = 'none';
        translationsTableBody.innerHTML = '';
        errorText.style.display = 'none';
        console.log("Result area reset.");
    }

    function resetToUploadState() {
        console.log("Resetting UI to initial upload state.");
        resetResultArea();
        resetFileSelection();
        progressSection.style.display = 'none';
        batchSummarySection.style.display = 'none';
        uploadSection.style.display = 'block';
        processButton.disabled = !(selectedFile && isConnected);
        processedImageCount = 0;
        totalImagesInBatch = 0;
        batchResultsContainer.innerHTML = '';
        processAnotherButton.style.display = 'none'; // Ensure this is hidden
    }

    // --- Initial Page Load State ---
    resetToUploadState();
    console.log("Initial UI state set. Waiting for connection and file selection.");

}); // End DOMContentLoaded
