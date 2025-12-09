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
    const loadingIndicator = document.getElementById('imageLoadingIndicator');
    const resultImage = document.getElementById('resultImage');
    const downloadLink = document.getElementById('downloadLink');
    const tableResultArea = document.getElementById('table-result-area');
    const translationsTableBody = document.getElementById('translationsTable').querySelector('tbody');
    const processAnotherButton = document.getElementById('processAnotherButton');
    const modeExtractRadio = document.getElementById('modeExtract');
    const modeAutoRadio = document.getElementById('modeAuto');
    
    // --- Elements for Batch Processing Results ---
    const batchResultContainer = document.getElementById('batch-result-container'); // Assume this new div exists
    const batchSummaryText = document.getElementById('batchSummaryText'); // Assume this new span/p exists
    const batchImagesList = document.getElementById('batchImagesList'); // Assume this new UL exists
    
    // --- State Variables ---
    let selectedFile = null;
    let isConnected = false;
    let isBatchProcessing = false; // Flag to track batch mode
    let batchTotalImages = 0;
    let batchCompletedImages = 0;

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
            console.log("   Process button enabled (reconnected/file selected).");
        } else {
            console.log("   Waiting for file selection.");
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
        if (!isBatchProcessing) {
            progressText.textContent = data.message || '⏳ بدأت المعالجة...';
            progressBar.value = 5;
        }
    });

    socket.on('progress_update', (data) => {
        if (!isBatchProcessing) {
            const percentage = (data.percentage >= 0 && data.percentage <= 100) ? data.percentage : progressBar.value;
            progressBar.value = percentage;
            const stepPrefix = data.step >= 0 ? `[${data.step}/6] ` : '';
            progressText.textContent = `${stepPrefix}${data.message} (${percentage}%)`;
            errorText.style.display = 'none';
        } else {
             progressText.textContent = `⏳ معالجة دفعة (${batchCompletedImages}/${batchTotalImages}): ${data.message}`;
        }
    });
    
    // --- Batch Started Listener ---
    socket.on('batch_started', (data) => {
        isBatchProcessing = true;
        batchTotalImages = data.total_images;
        batchCompletedImages = 0;
        console.log(`Batch processing started for ${batchTotalImages} images.`);
        
        batchImagesList.innerHTML = '';
        batchSummaryText.textContent = `بدء معالجة ${batchTotalImages} صورة...`;
        
        progressBar.value = 5;
        progressText.textContent = `⏳ معالجة دفعة: جاري إطلاق ${batchTotalImages} مهمة...`;
        
        imageResultArea.style.display = 'none';
        tableResultArea.style.display = 'none';
        
        resultSection.style.display = 'block';
        batchResultContainer.style.display = 'block';
    });


    socket.on('processing_complete', (data) => {
        console.log('✅ Processing complete! Data:', data);

        // --- Handle Batch Completion ---
        if (data.is_zip_batch) {
             batchCompletedImages++;
             
             const batchProgress = Math.round((batchCompletedImages / batchTotalImages) * 100);
             progressBar.value = batchProgress;
             
             batchSummaryText.textContent = `جاري معالجة: ${batchCompletedImages} من ${batchTotalImages} صورة (${batchProgress}%)`;

             const listItem = document.createElement('li');
             const modeText = data.mode === 'extract' ? ' (تنظيف/استخراج)' : ' (ترجمة تلقائية)';
             const originalName = data.original_filename || 'unknown';
             
             const link = document.createElement('a');
             link.href = data.imageUrl;
             link.target = '_blank';
             link.download = generateDownloadFilename(originalName, data.mode === 'auto' ? '_translated' : '_cleaned');
             link.textContent = `✔️ ${originalName} ${modeText}`;
             
             listItem.appendChild(link);
             
             if (data.mode === 'extract' && data.translations && data.translations.length > 0) {
                 const tableLink = document.createElement('span');
                 tableLink.textContent = ' [عرض الترجمات]';
                 tableLink.style.cursor = 'pointer';
                 tableLink.style.color = '#007bff';
                 tableLink.onclick = () => showTranslationsModal(originalName, data.translations);
                 listItem.appendChild(tableLink);
             }
             
             batchImagesList.appendChild(listItem);

             if (batchCompletedImages === batchTotalImages) {
                 progressText.textContent = '✨ اكتملت معالجة الدفعة بالكامل!';
                 console.log("Batch fully completed.");
             }
             
             downloadLink.href = data.imageUrl;
             
             return; 
        }
        
        // --- Handle Single Image Completion (Original Logic) ---
        progressBar.value = 100;
        progressText.textContent = '✨ اكتملت المعالجة! جارٍ تحميل صورة النتيجة...';

        setTimeout(() => {
             progressSection.style.display = 'none';
        }, 500);

        resultSection.style.display = 'block';
        imageResultArea.style.display = 'none';
        tableResultArea.style.display = 'none';
        batchResultContainer.style.display = 'none';
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
            resultSection.style.display = 'none';
            return;
        }

        let baseDownloadName = generateDownloadFilename(data.original_filename || selectedFile?.name, "");
        let suffix = '';
        if (data.mode === 'extract' || data.mode === 'white_fill') {
            console.log("   Preparing 'extract' or 'white_fill' results.");
            imageResultTitle.textContent = data.mode === 'white_fill' ? "الصورة المنظفة (بالأبيض)" : "الصورة المنظفة/المستخلصة";
            suffix = data.mode === 'white_fill' ? '_cleaned_white.jpg' : '_cleaned.jpg';
            if (data.mode === 'extract' && data.translations) {
                 populateTable(data.translations);
                 tableResultArea.style.display = 'block';
            }
        } else if (data.mode === 'auto') {
            console.log("   Preparing 'auto' results.");
            imageResultTitle.textContent = "الصورة المترجمة تلقائياً";
            suffix = "_translated.jpg";
        }
        
        downloadLink.download = baseDownloadName.replace('.jpg', suffix);
        imageResultArea.style.display = 'block';
        downloadLink.href = data.imageUrl;

        // --- Handle actual image loading ---
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

        console.log("   Setting result image src:", data.imageUrl);
        resultImage.src = data.imageUrl + '?t=' + Date.now();
    });

    socket.on('processing_error', (data) => {
        console.error('❌ Processing Error:', data.error);
        
        // Handle batch error display
        if (isBatchProcessing) {
             const listItem = document.createElement('li');
             listItem.style.color = 'red';
             listItem.textContent = `❌ خطأ في معالجة الملف: ${data.original_filename || 'غير محدد'} - ${data.error}`;
             batchImagesList.appendChild(listItem);
             batchCompletedImages++;
             
             const batchProgress = Math.round((batchCompletedImages / batchTotalImages) * 100);
             progressBar.value = batchProgress;
             batchSummaryText.textContent = `جاري معالجة: ${batchCompletedImages} من ${batchTotalImages} صورة (${batchProgress}%)`;
             
             if (batchCompletedImages === batchTotalImages) {
                 progressText.textContent = '⚠️ اكتملت معالجة الدفعة مع بعض الأخطاء.';
             }
             return;
        }
        
        // Handle single image error display
        errorText.textContent = `😭 خطأ في المعالجة: ${data.error}`;
        errorText.style.display = 'block';
        progressSection.style.display = 'none';
        resultSection.style.display = 'none';
        uploadSection.style.display = 'block';
        processButton.disabled = !(selectedFile && isConnected);
    });

    // --- DOM Event Listeners ---
    imageUpload.addEventListener('change', (event) => {
        resetResultArea();
        errorText.style.display = 'none';

        selectedFile = event.target.files[0];
        console.log("File selected:", selectedFile);
        if (selectedFile) {
             const allowedTypes = ['image/png', 'image/jpeg', 'image/webp', 'application/zip'];
             const maxZipSizeMB = 50; 
             const maxZipSizeBytes = maxZipSizeMB * 1024 * 1024;
             
             const fileType = selectedFile.type === '' && selectedFile.name.toLowerCase().endsWith('.zip') ? 'application/zip' : selectedFile.type;
             
             let currentMaxSize = 9999999999999; 
             if (fileType === 'application/zip') {
                  currentMaxSize = maxZipSizeBytes; 
             }
             
             // Validate Type
             if (!allowedTypes.includes(fileType)) {
                 alert(`نوع الملف غير صالح: ${fileType || selectedFile.name}.\nالأنواع المسموحة: PNG, JPG, WEBP, ZIP.`);
                 resetFileSelection(); return;
             }
             // Validate Size
             if (selectedFile.size > currentMaxSize) {
                 alert(`حجم الملف كبير جدًا (${(selectedFile.size / 1024 / 1024).toFixed(1)} MB).\nالحد الأقصى: ${currentMaxSize / 1024 / 1024} MB.`);
                 resetFileSelection(); return;
             }

             fileNameSpan.textContent = selectedFile.name;
             processButton.disabled = !isConnected;
             if (!isConnected) { console.warn("Socket not connected yet, process button disabled."); }

        } else {
            resetFileSelection();
        }
    });

    // ❌ تم إزالة e.preventDefault(); هنا لحل مشكلة إغلاق النافذة المفاجئ 
    fileUploadLabel.addEventListener('click', (e) => {
        // e.preventDefault(); // تم إزالة هذا السطر
        imageUpload.click();
    });

    // --- Process Button Click Handler (MODIFIED for ZIP) ---
    processButton.addEventListener('click', () => {
        console.log("Process button clicked.");
        if (!selectedFile) { alert('الرجاء اختيار ملف صورة أو ملف مضغوط أولاً.'); return; }
        if (!isConnected) { alert('لا يوجد اتصال بالخادم. الرجاء الانتظار أو تحديث الصفحة.'); return; }

        const currentMode = modeAutoRadio.checked ? 'auto' : 'extract';
        const isZipFile = selectedFile.name.toLowerCase().endsWith('.zip');
        const uploadEndpoint = isZipFile ? '/upload_zip' : '/upload';
        
        console.log(`   Mode selected: ${currentMode}, File type: ${isZipFile ? 'ZIP' : 'Image'}, Endpoint: ${uploadEndpoint}`);

        uploadSection.style.display = 'none';
        progressSection.style.display = 'block';
        resultSection.style.display = 'none';
        errorText.style.display = 'none';
        progressBar.value = 0;
        progressText.textContent = `⏫ بدء الرفع... (0%) ${isZipFile ? '[ملف مضغوط]' : ''}`;
        processButton.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);

        const xhr = new XMLHttpRequest();

        // --- Progress Event Listener ---
        xhr.upload.addEventListener('progress', (event) => {
            if (event.lengthComputable) {
                const percentage = Math.round((event.loaded / event.total) * 100);
                progressBar.value = percentage;
                progressText.textContent = `⏫ جارٍ رفع الملف... (${percentage}%)`;
            } else {
                progressText.textContent = '⏫ جارٍ رفع الملف... (الحجم غير محدد)';
            }
        }, false);

        // --- Load Event Listener (Upload Complete/Server Responded) ---
        xhr.addEventListener('load', () => {
            console.log(`   XHR Upload finished with status: ${xhr.status}`);

            let resultJson;
            try {
                if (!xhr.responseText) { throw new Error("استجابة فارغة من الخادم."); }
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

                if (isZipFile) {
                    const { images_to_process } = resultJson;
                    if (!images_to_process || images_to_process.length === 0) {
                         console.error("   ❌ ZIP had no images to process:", resultJson);
                         errorText.textContent = "😭 خطأ: لم يتم العثور على صور صالحة في الملف المضغوط.";
                         errorText.style.display = 'block';
                         resetUiAfterError(true);
                         return;
                    }
                    
                    socket.emit('start_batch_processing', {
                        images_to_process: images_to_process,
                        mode: currentMode
                    });
                    console.log(`   ✅ Emitted 'start_batch_processing' for ${images_to_process.length} images.`);
                    
                } else {
                    const { output_filename_base, saved_filename } = resultJson;
                    if (!output_filename_base || !saved_filename) {
                        console.error("   ❌ Incomplete data from server:", resultJson);
                        errorText.textContent = "😭 خطأ: بيانات ملف غير مكتملة من الخادم بعد الرفع.";
                        errorText.style.display = 'block';
                        resetUiAfterError(true);
                        return;
                    }

                    socket.emit('start_processing', {
                        output_filename_base: output_filename_base,
                        saved_filename: saved_filename,
                        mode: currentMode
                    });
                    console.log("   ✅ Emitted 'start_processing' via SocketIO.");
                }

            } else {
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
            resetUiAfterError(true);
         });


        // --- Open and Send the Request ---
        try {
             console.log(`   Opening and sending XHR POST request to ${uploadEndpoint}...`);
             xhr.open('POST', uploadEndpoint, true);
             xhr.send(formData);
        } catch (sendError) {
             console.error("   ❌ Error initiating XHR send:", sendError);
             errorText.textContent = `😭 خطأ غير متوقع عند محاولة رفع الملف.`;
             errorText.style.display = 'block';
             resetUiAfterError(true);
        }
    });

    // --- Process Another Button ---
    processAnotherButton.addEventListener('click', () => {
        console.log("Process Another clicked.");
        resetToUploadState();
    });

    // --- Helper Function to Reset UI after Upload/Processing Error ---
    function resetUiAfterError(allowRetry = true) {
         isBatchProcessing = false;
         batchTotalImages = 0;
         batchCompletedImages = 0;
         progressSection.style.display = 'none';
         uploadSection.style.display = 'block';
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
            cellText.innerHTML = safeText.replace(/</g, "&lt;")
                                        .replace(/>/g, "&gt;")
                                        .replace(/\n/g, '<br>');
        });
    }
    
    // --- Function to display translations in a modal (simplified example using alert) ---
    function showTranslationsModal(filename, translations) {
        let text = `الترجمات المستخرجة لملف: ${filename}\n\n`;
        translations.forEach(item => {
             text += `[${item.id || '-'}] ${item.translation}\n---\n`;
        });
        alert(text);
        console.log(`Translations for ${filename} displayed.`);
    }


    function generateDownloadFilename(originalName, suffix) {
        const defaultName = "processed_image";
        let baseName = defaultName;
        let originalExtension = '.jpg'; 

        if (originalName && typeof originalName === 'string') {
            const lastDotIndex = originalName.lastIndexOf('.');
            if (lastDotIndex > 0) {
                 baseName = originalName.substring(0, lastDotIndex);
                 originalExtension = originalName.substring(lastDotIndex).toLowerCase();
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
        console.log("File selection reset.");
    }

    function resetResultArea() {
        resultSection.style.display = 'none';
        imageResultArea.style.display = 'none';
        tableResultArea.style.display = 'none';
        batchResultContainer.style.display = 'none';
        batchImagesList.innerHTML = '';
        batchSummaryText.textContent = '';
        
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
        isBatchProcessing = false;
        batchTotalImages = 0;
        batchCompletedImages = 0;
        resetResultArea();
        resetFileSelection();
        progressSection.style.display = 'none';
        uploadSection.style.display = 'block';
        processButton.disabled = !(selectedFile && isConnected);
    }

    // --- Initial Page Load State ---
    resetToUploadState();
    console.log("Initial UI state set. Waiting for connection and file selection.");

}); // End DOMContentLoaded
