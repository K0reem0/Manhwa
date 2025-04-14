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
    // Assumes Socket.IO library is loaded in your HTML
    // Connects to the same server serving the page by default
    const socket = io({
        // Optional: Add reconnection options if needed
        // reconnectionAttempts: 5,
        // reconnectionDelay: 1000,
    });

    // --- SocketIO Event Listeners ---
    socket.on('connect', () => {
        isConnected = true;
        console.log('✅ Socket.IO connected! SID:', socket.id);
        if (selectedFile) {
            processButton.disabled = false; // Enable button if file already selected
            console.log("   Process button enabled (file was selected).");
        } else {
            console.log("   Waiting for file selection.");
            processButton.disabled = true; // Ensure button is disabled if no file
        }
    });
    socket.on('disconnect', (reason) => {
        isConnected = false;
        console.warn('❌ Socket.IO disconnected! Reason:', reason);
        processButton.disabled = true; // Disable button on disconnect
        // Avoid alert spam if it's a brief disconnect/reconnect
        if (reason !== 'io server disconnect') { // Allow server-initiated disconnect without alert
             alert("⚠️ تم قطع الاتصال بالخادم. قد تحتاج لتحديث الصفحة. Reason: " + reason);
        }
        // Don't reset state immediately, allow reconnection attempts
        // resetToUploadState();
    });
    socket.on('connect_error', (error) => {
         isConnected = false;
         console.error('❌ Socket.IO connection error:', error);
         alert("❌ فشل الاتصال بالخادم. تأكد من تشغيل الخادم وحاول تحديث الصفحة.");
         processButton.disabled = true;
         resetToUploadState(); // Reset fully on connection error
    });

    // --- Processing Status Listeners (No change needed here) ---
    socket.on('processing_started', (data) => {
        console.log('Processing started:', data.message);
        progressText.textContent = data.message || '⏳ بدأت المعالجة...';
        progressBar.value = 5; // Initial small progress
    });
    socket.on('progress_update', (data) => {
        progressBar.value = data.percentage;
        const stepPrefix = data.step >= 0 ? `[${data.step}/6] ` : ''; // Handle step -1 nicely
        progressText.textContent = `${stepPrefix}${data.message} (${data.percentage}%)`;
        errorText.style.display = 'none'; // Hide previous errors on progress
    });
    socket.on('processing_complete', (data) => {
        console.log('✅ Processing complete! Data:', data);
        progressText.textContent = '✨ اكتملت المعالجة!';
        progressBar.value = 100;
        progressSection.style.display = 'none'; // Hide progress bar
        resultSection.style.display = 'block'; // Show results
        imageResultArea.style.display = 'none'; // Reset visibility
        tableResultArea.style.display = 'none'; // Reset visibility
        translationsTableBody.innerHTML = ''; // Clear old table data

        // Validate received data
        if (!data || !data.mode || !data.imageUrl) {
            console.error("Invalid data received on completion", data);
            errorText.textContent = "خطأ: بيانات نتيجة غير صالحة من الخادم.";
            errorText.style.display = 'block';
             // Offer reset
             uploadSection.style.display = 'block';
             processButton.disabled = !selectedFile; // Re-enable if file still selected
            return;
        }

        // Display results based on mode
        if (data.mode === 'extract') {
            console.log("   Displaying 'extract' results.");
            imageResultTitle.textContent = "الصورة المنظفة";
            resultImage.src = data.imageUrl + '?t=' + Date.now(); // Cache bust
            downloadLink.href = data.imageUrl;
            downloadLink.download = generateDownloadFilename(selectedFile?.name, "_cleaned");
            imageResultArea.style.display = 'block'; // Show image result

            populateTable(data.translations); // Populate table with translations
            tableResultArea.style.display = 'block'; // Show table area
        } else if (data.mode === 'auto') {
            console.log("   Displaying 'auto' results.");
            imageResultTitle.textContent = "الصورة المترجمة تلقائياً";
            resultImage.src = data.imageUrl + '?t=' + Date.now(); // Cache bust
            downloadLink.href = data.imageUrl;
            downloadLink.download = generateDownloadFilename(selectedFile?.name, "_translated");
            imageResultArea.style.display = 'block'; // Show image result
            // No table for auto mode in this example
        }
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
        // Reset previous state first
        resetResultArea();
        errorText.style.display = 'none'; // Hide previous errors

        selectedFile = event.target.files[0];
        console.log("File selected:", selectedFile);
        if (selectedFile) {
             const allowedTypes = ['image/png', 'image/jpeg', 'image/webp'];
             const maxSizeMB = 5000; // Match Flask config
             const maxSizeBytes = maxSizeMB * 1024 * 1024;

             // Validate Type
             if (!allowedTypes.includes(selectedFile.type)) {
                 alert(`نوع الملف غير صالح: ${selectedFile.type}.\nالأنواع المسموحة: PNG, JPG, WEBP.`);
                 resetFileSelection();
                 return;
             }
             // Validate Size
             if (selectedFile.size > maxSizeBytes) {
                 alert(`حجم الملف كبير جدًا (${(selectedFile.size / 1024 / 1024).toFixed(1)} MB).\nالحد الأقصى: ${maxSizeMB} MB.`);
                 resetFileSelection();
                 return;
             }

             fileNameSpan.textContent = selectedFile.name; // Show filename
             processButton.disabled = !isConnected; // Enable button only if connected
             if (!isConnected) { console.warn("Socket not connected yet, process button disabled."); }

        } else {
            resetFileSelection(); // Clear selection if no file chosen
        }
    });

    // Allow clicking the label to trigger the hidden file input
    fileUploadLabel.addEventListener('click', (e) => {
        e.preventDefault(); // Prevent label's default behavior if any
        imageUpload.click(); // Trigger the file input click
    });

    // === MODIFIED: Process Button Click Handler ===
    processButton.addEventListener('click', async () => { // <-- Make the handler async
        console.log("Process button clicked.");
        if (!selectedFile) { alert('الرجاء اختيار ملف صورة أولاً.'); return; }
        if (!isConnected) { alert('لا يوجد اتصال بالخادم. الرجاء الانتظار أو تحديث الصفحة.'); return; }

        const currentMode = modeAutoRadio.checked ? 'auto' : 'extract';
        console.log(`   Mode selected: ${currentMode}`);

        // --- Update UI for Uploading State ---
        uploadSection.style.display = 'none'; // Hide upload controls
        progressSection.style.display = 'block'; // Show progress section
        resultSection.style.display = 'none'; // Hide previous results
        errorText.style.display = 'none'; // Hide previous errors
        progressBar.value = 0;
        progressText.textContent = '⏫ جارٍ رفع الصورة...'; // Initial message for upload
        processButton.disabled = true; // Disable button during processing

        // --- Create FormData for POST request ---
        const formData = new FormData();
        formData.append('file', selectedFile); // Key 'file' MUST match backend Flask route (request.files['file'])

        // --- Perform the POST request to /upload ---
        try {
            console.log("   Sending POST request to /upload...");
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
                // Headers are automatically set by browser for FormData with files
            });

            // Always expect JSON, even for errors from backend /upload route
            const result = await response.json();

            // Check if upload was successful on the server
            if (!response.ok) {
                // Throw an error with the message from the server's JSON response
                throw new Error(result.error || `فشل الرفع (خطأ ${response.status})`);
            }

            // --- Upload successful, now emit SocketIO event ---
            console.log("   ✅ POST Upload successful:", result);
            progressText.textContent = '⏳ تم الرفع، جارٍ بدء المعالجة...'; // Update progress text

            // Extract necessary info from the successful upload response
            const { output_filename_base, saved_filename } = result;
            if (!output_filename_base || !saved_filename) {
                 throw new Error("بيانات ملف غير مكتملة من الخادم بعد الرفع.");
            }

            // Emit 'start_processing' with the received identifiers
            socket.emit('start_processing', {
                output_filename_base: output_filename_base,
                saved_filename: saved_filename, // Send the exact name back
                mode: currentMode
            });
            console.log("   ✅ Emitted 'start_processing' via SocketIO.");

            // UI remains in progress state, waiting for SocketIO updates
            // Button remains disabled

        } catch (error) {
            // Handle network errors during fetch or errors thrown from response handling
            console.error("   ❌ Error during upload or triggering processing:", error);
            errorText.textContent = `😭 خطأ: ${error.message}`;
            errorText.style.display = 'block';

            // Reset UI to allow retry, but keep file selected
            progressSection.style.display = 'none'; // Hide progress
            uploadSection.style.display = 'block'; // Show upload controls again
            // Re-enable button only if connected
            processButton.disabled = !isConnected;
            // DO NOT call resetFileSelection() here, user might want to retry with same file
        }
    });

    // Reset button logic (no change needed)
    processAnotherButton.addEventListener('click', () => {
        console.log("Process Another clicked.");
        resetToUploadState();
    });

    // --- Helper Functions (No changes needed) ---
    function populateTable(translations) {
        translationsTableBody.innerHTML = ''; // Clear previous entries
        if (!translations || translations.length === 0) {
            const row = translationsTableBody.insertRow();
            const cell = row.insertCell();
            cell.colSpan = 2; // Span across both columns
            cell.textContent = "لم يتم استخراج أي نصوص أو ترجمات.";
            cell.style.textAlign = 'center';
            return;
        }
        translations.forEach(item => {
            const row = translationsTableBody.insertRow();
            const cellId = row.insertCell();
            const cellText = row.insertCell();
            // Ensure ID is displayed, default to '-' if missing
            cellId.textContent = (item.id !== undefined && item.id !== null) ? item.id : '-';
            // Ensure text is string, escape HTML, handle newlines
            const safeText = item.translation ? String(item.translation) : '';
            cellText.innerHTML = safeText.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, '<br>');
        });
    }
    function generateDownloadFilename(originalName, suffix) {
        const defaultName = "processed_image";
        let baseName = defaultName;
        // Try to get base name from original, handle cases where it might be missing/weird
        if (originalName && typeof originalName === 'string') {
            baseName = originalName.split('.').slice(0, -1).join('.') || defaultName;
        }
        // Ensure suffix is added, default to jpg if backend might save differently unexpectedly
        return `${baseName}${suffix || ''}.jpg`;
    }
    function resetFileSelection() {
        imageUpload.value = null; // Clear the file input
        selectedFile = null;
        fileNameSpan.textContent = 'لم يتم اختيار أي ملف';
        processButton.disabled = true; // Disable button when no file
        console.log("File selection reset.");
    }
    function resetResultArea() {
        resultSection.style.display = 'none';
        imageResultArea.style.display = 'none';
        tableResultArea.style.display = 'none';
        resultImage.src = "#"; // Clear image source
        downloadLink.href = "#";
        translationsTableBody.innerHTML = ''; // Clear table
        errorText.style.display = 'none'; // Hide error text
        console.log("Result area reset.");
    }
    function resetToUploadState() {
        console.log("Resetting UI to initial upload state.");
        resetResultArea(); // Clear results
        resetFileSelection(); // Clear file selection
        progressSection.style.display = 'none'; // Hide progress
        uploadSection.style.display = 'block'; // Show upload
        // Button state is handled by file selection and connection status checks
    }

    // --- Initial State ---
    resetToUploadState(); // Set initial UI state on page load
    console.log("Initial UI state set.");
});
