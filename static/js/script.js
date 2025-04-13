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
    const socket = io();

    // --- SocketIO Event Listeners ---
    socket.on('connect', () => {
        isConnected = true; console.log('✅ Socket.IO connected! SID:', socket.id);
        if (selectedFile) { processButton.disabled = false; console.log("   Process button enabled."); }
        else { console.log("   Waiting for file selection."); }
    });
    socket.on('disconnect', (reason) => {
        isConnected = false; console.warn('❌ Socket.IO disconnected! Reason:', reason); processButton.disabled = true;
        alert("⚠️ تم قطع الاتصال بالخادم. Reason: " + reason); resetToUploadState();
    });
    socket.on('connect_error', (error) => {
         isConnected = false; console.error('❌ Socket.IO connection error:', error);
         alert("❌ فشل الاتصال بالخادم."); processButton.disabled = true; resetToUploadState();
    });
    socket.on('processing_started', (data) => {
        console.log('Processing started:', data.message); progressText.textContent = data.message; progressBar.value = 5;
    });
    socket.on('progress_update', (data) => {
        progressBar.value = data.percentage; const stepPrefix = data.step ? `[${data.step}/6] ` : '';
        progressText.textContent = `${stepPrefix}${data.message} (${data.percentage}%)`; errorText.style.display = 'none';
    });
    socket.on('processing_complete', (data) => {
        console.log('✅ Processing complete! Data:', data); progressText.textContent = '✨ اكتملت المعالجة!'; progressBar.value = 100;
        progressSection.style.display = 'none'; resultSection.style.display = 'block';
        imageResultArea.style.display = 'none'; tableResultArea.style.display = 'none'; translationsTableBody.innerHTML = '';

        if (!data || !data.mode || !data.imageUrl) { console.error("Invalid data received", data); errorText.textContent = "خطأ: بيانات نتيجة غير صالحة."; errorText.style.display = 'block'; return; }

        if (data.mode === 'extract') {
            console.log("   Displaying 'extract' results."); imageResultTitle.textContent = "الصورة المنظفة";
            resultImage.src = data.imageUrl + '?t=' + Date.now(); downloadLink.href = data.imageUrl;
            downloadLink.download = generateDownloadFilename(selectedFile?.name, "_cleaned"); imageResultArea.style.display = 'block';
            populateTable(data.translations); tableResultArea.style.display = 'block'; // Show table even if empty
        } else if (data.mode === 'auto') {
            console.log("   Displaying 'auto' results."); imageResultTitle.textContent = "الصورة المترجمة تلقائياً";
            resultImage.src = data.imageUrl + '?t=' + Date.now(); downloadLink.href = data.imageUrl;
            downloadLink.download = generateDownloadFilename(selectedFile?.name, "_translated"); imageResultArea.style.display = 'block';
        }
    });
    socket.on('processing_error', (data) => {
        console.error('❌ Processing Error:', data.error); errorText.textContent = `😭 خطأ: ${data.error}`; errorText.style.display = 'block';
        progressSection.style.display = 'block'; progressBar.value = 0; resultSection.style.display = 'none';
        uploadSection.style.display = 'block'; processButton.disabled = false;
    });

    // --- DOM Event Listeners ---
    imageUpload.addEventListener('change', (event) => {
        selectedFile = event.target.files[0]; console.log("File selected:", selectedFile);
        if (selectedFile) {
             const allowedTypes = ['image/png', 'image/jpeg', 'image/webp'];
             if (!allowedTypes.includes(selectedFile.type)) { alert(`نوع الملف غير صالح: ${selectedFile.type}.`); resetFileSelection(); return; }
             if (selectedFile.size > 16 * 1024 * 1024) { alert(`حجم الملف كبير جدًا (${(selectedFile.size / 1024 / 1024).toFixed(1)} MB).`); resetFileSelection(); return; }
             fileNameSpan.textContent = selectedFile.name; processButton.disabled = !isConnected;
             if (!isConnected) { console.warn("Socket not connected yet."); } resetResultArea();
        } else { resetFileSelection(); }
    });
     fileUploadLabel.addEventListener('click', (e) => { e.preventDefault(); imageUpload.click(); });
     processButton.addEventListener('click', () => {
        console.log("Process button clicked.");
        if (!selectedFile || !isConnected) { alert(!selectedFile ? 'اختر ملف أولاً.' : 'لا يوجد اتصال بالخادم.'); return; }
        const currentMode = modeAutoRadio.checked ? 'auto' : 'extract'; console.log(`   Mode: ${currentMode}`);
        uploadSection.style.display = 'none'; progressSection.style.display = 'block'; resultSection.style.display = 'none';
        errorText.style.display = 'none'; progressBar.value = 0; progressText.textContent = '⏳ جارٍ قراءة الملف...'; processButton.disabled = true;
        const reader = new FileReader();
        reader.onload = function(event) {
            try {
                const base64String = event.target.result; console.log("   FileReader loaded.");
                if (!base64String || !base64String.startsWith('data:image')) { throw new Error("Invalid Data URL."); }
                console.log(`   Emitting 'start_processing' (Mode: ${currentMode})...`); progressText.textContent = '⏫ جارٍ رفع الصورة...';
                socket.emit('start_processing', { file: base64String, mode: currentMode }); console.log("   ✅ Event emitted.");
            } catch (error) { console.error("   ❌ Error processing/emitting:", error); alert("خطأ تجهيز الملف: " + error.message); resetToUploadState(); }
        };
        reader.onerror = function(error) { console.error("   ❌ FileReader error:", error); alert("خطأ قراءة الملف: " + error.message); resetToUploadState(); };
        console.log("   Reading file..."); reader.readAsDataURL(selectedFile);
    });
     processAnotherButton.addEventListener('click', () => { console.log("Process Another clicked."); resetToUploadState(); });

    // --- Helper Functions ---
    function populateTable(translations) {
        translationsTableBody.innerHTML = ''; if (!translations || translations.length === 0) { const row = translationsTableBody.insertRow(); const cell = row.insertCell(); cell.colSpan = 2; cell.textContent = "لم يتم استخراج أي نصوص."; cell.style.textAlign = 'center'; return; }
        translations.forEach(item => { const row = translationsTableBody.insertRow(); const cellId = row.insertCell(); const cellText = row.insertCell(); cellId.textContent = item.id !== undefined ? item.id : '-'; const safeText = item.translation ? String(item.translation) : ''; cellText.innerHTML = safeText.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, '<br>'); });
    }
    function generateDownloadFilename(originalName, suffix) { const defaultName = "processed_image"; let baseName = defaultName; if (originalName && typeof originalName === 'string') { baseName = originalName.split('.').slice(0, -1).join('.') || defaultName; } return `${baseName}${suffix}.jpg`; }
    function resetFileSelection() { imageUpload.value = null; selectedFile = null; fileNameSpan.textContent = 'لم يتم اختيار أي ملف'; processButton.disabled = true; console.log("File selection reset."); }
    function resetResultArea() { resultSection.style.display = 'none'; imageResultArea.style.display = 'none'; tableResultArea.style.display = 'none'; resultImage.src = "#"; downloadLink.href = "#"; translationsTableBody.innerHTML = ''; errorText.style.display = 'none'; console.log("Result area reset."); }
    function resetToUploadState() { console.log("Resetting UI state."); resetResultArea(); resetFileSelection(); progressSection.style.display = 'none'; uploadSection.style.display = 'block'; }
    resetToUploadState(); console.log("Initial UI state set.");
});
