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
    const batchLog = document.getElementById('batch-log'); // عنصر جديد لسجل الحزمة
    const errorText = document.getElementById('errorText');
    
    const resultSection = document.getElementById('result-section');
    
    // Single Image Result Elements
    const imageResultArea = document.getElementById('image-result-area');
    const imageResultTitle = document.getElementById('image-result-title');
    const resultImage = document.getElementById('resultImage');
    const downloadLink = document.getElementById('downloadLink');
    
    // Batch Result Elements
    const batchResultArea = document.getElementById('batch-result-area'); // عنصر جديد
    const batchLinksList = document.getElementById('batch-links-list'); // عنصر جديد

    const processAnotherButton = document.getElementById('processAnotherButton');

    let selectedFile = null;
    let isConnected = false;
    let isBatchMode = false; // لتتبع حالة المعالجة الحالية

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
        progressText.textContent = `⚠️ جاري محاولة إعادة الاتصال (${attempt})...`;
    });

    socket.on('connect_error', (error) => {
         isConnected = false;
         console.error('❌ Socket.IO connection error:', error);
         errorText.textContent = "❌ فشل الاتصال بالخادم.";
         errorText.style.display = 'block';
         processButton.disabled = true;
    });

    // --- SocketIO Processing Listeners ---
    
    // 1. استقبال بداية الحزمة (خاص بملفات ZIP)
    socket.on('batch_started', (data) => {
        console.log('Batch started:', data);
        const msg = document.createElement('div');
        msg.textContent = `🚀 تم استلام ${data.total_images} صورة. بدء المعالجة...`;
        msg.style.color = 'blue';
        if(batchLog) batchLog.appendChild(msg);
    });

    // 2. تحديث التقدم
    socket.on('progress_update', (data) => {
        const percentage = (data.percentage >= 0 && data.percentage <= 100) ? data.percentage : progressBar.value;
        progressBar.value = percentage;
        progressText.textContent = `${data.message} (${percentage}%)`;
        errorText.style.display = 'none';
    });

    // 3. اكتمال المعالجة (يتم استدعاؤها لكل صورة على حدة)
    socket.on('processing_complete', (data) => {
        console.log('✅ Processing complete for item:', data);
        
        // إظهار قسم النتائج
        resultSection.style.display = 'block';
        
        // إضافة طابع زمني لتجنب الكاش
        const finalUrl = data.imageUrl + '?t=' + Date.now();

        if (isBatchMode) {
            // --- وضع الحزمة (ZIP) ---
            batchResultArea.style.display = 'block';
            imageResultArea.style.display = 'none'; // إخفاء عرض الصورة الفردية
            progressSection.style.display = 'block'; // إبقاء شريط التقدم ظاهراً

            // إضافة رابط للملف المعالج في القائمة
            const li = document.createElement('li');
            li.className = "batch-item"; // يمكن تنسيق هذا في CSS
            li.style.marginBottom = "8px";
            li.innerHTML = `
                <span>📄 ${data.original_filename}</span> 
                <span style="margin: 0 10px;">➔</span>
                <a href="${finalUrl}" target="_blank" class="btn btn-sm" style="padding: 2px 8px; font-size: 0.8em;">عرض</a>
                <a href="${finalUrl}" download class="btn btn-sm btn-primary" style="padding: 2px 8px; font-size: 0.8em;">تحميل</a>
            `;
            batchLinksList.appendChild(li);

            // تحديث السجل
            if(batchLog) {
                const logMsg = document.createElement('div');
                logMsg.textContent = `✔️ تم: ${data.original_filename}`;
                logMsg.style.color = "green";
                batchLog.appendChild(logMsg);
                batchLog.scrollTop = batchLog.scrollHeight; // تمرير لأسفل
            }

        } else {
            // --- وضع الصورة الفردية ---
            progressBar.value = 100;
            progressText.textContent = '✨ اكتملت المعالجة!';
            
            // إخفاء شريط التقدم بعد فترة قصيرة
            setTimeout(() => { progressSection.style.display = 'none'; }, 500);

            imageResultArea.style.display = 'block';
            batchResultArea.style.display = 'none';
            
            imageResultTitle.textContent = "الصورة المعالجة";
            downloadLink.href = finalUrl;
            downloadLink.download = "cleaned_" + data.original_filename;
            downloadLink.style.display = 'inline-block';

            // تحميل الصورة
            resultImage.onload = () => {
                resultImage.style.display = 'block';
            };
            resultImage.src = finalUrl;
        }
    });

    socket.on('processing_error', (data) => {
        console.error('❌ Processing Error:', data.error);
        if (isBatchMode && batchLog) {
            // في وضع الحزمة، نسجل الخطأ في السجل بدلاً من إيقاف كل شيء
            const errDiv = document.createElement('div');
            errDiv.textContent = `❌ خطأ: ${data.error}`;
            errDiv.style.color = 'red';
            batchLog.appendChild(errDiv);
        } else {
            errorText.textContent = `😭 خطأ في المعالجة: ${data.error}`;
            errorText.style.display = 'block';
            progressSection.style.display = 'none';
            processButton.disabled = !(selectedFile && isConnected);
        }
    });

    // --- DOM Event Listeners ---
    imageUpload.addEventListener('change', (event) => {
        resetResultArea();
        errorText.style.display = 'none';
        selectedFile = event.target.files[0];

        if (selectedFile) {
             // السماح بالصور والملفات المضغوطة
             const allowedTypes = [
                 'image/png', 'image/jpeg', 'image/webp', 'image/jpg',
                 'application/zip', 'application/x-zip-compressed', 'application/octet-stream'
             ];
             
             // التحقق البسيط من الامتداد لأن بعض المتصفحات لا تعطي MIME type دقيق لملفات zip
             const fileName = selectedFile.name.toLowerCase();
             const isZip = fileName.endsWith('.zip');
             const isImage = fileName.endsWith('.jpg') || fileName.endsWith('.jpeg') || fileName.endsWith('.png') || fileName.endsWith('.webp');

             if (!isZip && !isImage) {
                 alert(`نوع الملف غير مدعوم. يرجى رفع صورة (JPG, PNG) أو ملف مضغوط (ZIP).`);
                 resetFileSelection(); return;
             }

             fileNameSpan.textContent = selectedFile.name;
             processButton.disabled = !isConnected;
             
             // تحديد الوضع بناءً على الملف
             isBatchMode = isZip; 
             console.log(`File selected. Mode: ${isBatchMode ? 'Batch (ZIP)' : 'Single Image'}`);

        } else {
            resetFileSelection();
        }
    });

    fileUploadLabel.addEventListener('click', (e) => {
        e.preventDefault();
        imageUpload.click();
    });

    // --- Process Button Click Handler ---
    processButton.addEventListener('click', () => {
        if (!selectedFile) { alert('الرجاء اختيار ملف أولاً.'); return; }
        if (!isConnected) { alert('لا يوجد اتصال بالخادم.'); return; }

        // إعداد الواجهة للرفع
        uploadSection.style.display = 'none';
        progressSection.style.display = 'block';
        resultSection.style.display = 'none';
        errorText.style.display = 'none';
        
        // تنظيف واجهة الحزمة
        if(batchLog) batchLog.innerHTML = '';
        if(batchLinksList) batchLinksList.innerHTML = '';

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
            }
        });

        xhr.addEventListener('load', () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                let resultJson;
                try {
                    resultJson = JSON.parse(xhr.responseText);
                } catch (e) {
                     handleUploadError("فشل قراءة استجابة الخادم."); return;
                }

                progressBar.value = 100; 
                progressText.textContent = '⏳ تم الرفع. بدء المعالجة...';

                // إرسال إشارة البدء بناءً على نوع الملف
                if (isBatchMode) {
                    // للدفعة (ZIP)
                    if (resultJson.images_to_process) {
                        socket.emit('start_batch_processing', {
                            images_to_process: resultJson.images_to_process,
                            mode: 'clean_white'
                        });
                    } else {
                        handleUploadError("لم يتم العثور على صور صالحة داخل ملف ZIP.");
                    }
                } else {
                    // للصورة الفردية
                    socket.emit('start_processing', {
                        output_filename_base: resultJson.output_filename_base,
                        saved_filename: resultJson.saved_filename,
                        mode: 'clean_white' // الوضع الوحيد المتاح الآن
                    });
                }

            } else {
                let msg = "حدث خطأ أثناء الرفع.";
                try { msg = JSON.parse(xhr.responseText).error; } catch(e){}
                handleUploadError(msg);
            }
        });

        xhr.addEventListener('error', () => {
            handleUploadError("خطأ في الشبكة أثناء الرفع.");
        });

        // تحديد الرابط بناءً على نوع الملف
        const uploadUrl = isBatchMode ? '/upload_zip' : '/upload';
        
        try {
             xhr.open('POST', uploadUrl, true);
             xhr.send(formData);
        } catch (sendError) {
             handleUploadError("خطأ غير متوقع عند الإرسال.");
        }
    });

    function handleUploadError(msg) {
        console.error(msg);
        errorText.textContent = `خطأ: ${msg}`;
        errorText.style.display = 'block';
        progressSection.style.display = 'none';
        uploadSection.style.display = 'block';
        processButton.disabled = !(selectedFile && isConnected);
    }

    // --- UI Helper Functions ---
    processAnotherButton.addEventListener('click', () => {
        resetToUploadState();
    });

    function resetFileSelection() {
        imageUpload.value = null;
        selectedFile = null;
        fileNameSpan.textContent = 'لم يتم اختيار أي ملف';
        processButton.disabled = true;
        isBatchMode = false;
    }

    function resetResultArea() {
        resultSection.style.display = 'none';
        imageResultArea.style.display = 'none';
        batchResultArea.style.display = 'none';
        
        resultImage.src = "#";
        resultImage.style.display = 'none';
        
        if(batchLinksList) batchLinksList.innerHTML = '';
        if(batchLog) batchLog.innerHTML = '';
        
        errorText.style.display = 'none';
    }

    function resetToUploadState() {
        resetResultArea();
        resetFileSelection();
        progressSection.style.display = 'none';
        uploadSection.style.display = 'block';
    }

    // Initialize
    resetToUploadState();
});
