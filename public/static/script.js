const fileInput = document.getElementById('fileInput');
const filenamePreview = document.getElementById('filenamePreview');
const dropzone = document.getElementById('dropzone');
const uploadForm = document.getElementById('uploadForm');
const runBtn = document.getElementById('runBtn');

if (fileInput) {
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      const f = fileInput.files[0];
      const sizeKB = (f.size / 1024).toFixed(1);
      filenamePreview.textContent = `Dipilih: ${f.name} (${sizeKB} KB)`;
    }
  });

  ['dragover', 'dragenter'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.style.borderColor = '#1F6F5C';
      dropzone.style.background = '#f7fbf9';
    });
  });
  ['dragleave', 'drop'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.style.borderColor = '';
      dropzone.style.background = '';
    });
  });
  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      const f = fileInput.files[0];
      const sizeKB = (f.size / 1024).toFixed(1);
      filenamePreview.textContent = `Dipilih: ${f.name} (${sizeKB} KB)`;
    }
  });
}

const manualTabs = document.querySelectorAll('.manual-tab');
if (manualTabs.length) {
  manualTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.manual-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.manual-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(tab.dataset.target).classList.add('active');
    });
  });
}

const promptText = document.getElementById('promptText');
document.querySelectorAll('.prompt-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    if (!promptText) return;
    promptText.value = chip.dataset.prompt || chip.textContent.trim();
    promptText.focus();
  });
});

const llmToggleBtn = document.getElementById('llmToggleBtn');
const llmKeyField = document.getElementById('llmKeyField');
if (llmToggleBtn) {
  llmToggleBtn.addEventListener('click', () => {
    llmKeyField.classList.toggle('is-open');
    llmToggleBtn.textContent = llmKeyField.classList.contains('is-open')
      ? '✨ Narasi AI (Gemini) aktif — klik untuk sembunyikan'
      : '✨ Aktifkan narasi AI (Gemini) — opsional';
  });
}

if (uploadForm) {
  uploadForm.addEventListener('submit', () => {
    if (fileInput.files.length === 0) return;
    uploadForm.classList.add('is-loading');
    runBtn.disabled = true;
    runBtn.querySelector('span').textContent = 'Menganalisis';
  });
}
