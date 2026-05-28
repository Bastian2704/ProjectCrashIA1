document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("upload-form");
  if (!form) return;

  const submitBtn = document.getElementById("submit-btn");
  const progressSection = document.getElementById("progress-section");
  const statusText = document.getElementById("status-text");
  const errorSection = document.getElementById("error-section");
  const errorMsg = document.getElementById("error-msg");
  const modeHidden = document.getElementById("mode-hidden");
  const modelHidden = document.getElementById("model-hidden");

  // Sync hidden fields with radio selection
  document.querySelectorAll("input[name='mode-selector']").forEach((radio) => {
    radio.addEventListener("change", () => {
      const val = radio.value;
      if (val === "compare") {
        modeHidden.value = "compare";
        modelHidden.value = "n";
        submitBtn.textContent = "Comparar modelos";
      } else {
        modeHidden.value = "single";
        modelHidden.value = val;
        submitBtn.textContent = "Analizar video";
      }
    });
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    errorSection.classList.add("d-none");
    progressSection.classList.remove("d-none");
    submitBtn.disabled = true;

    const isCompare = modeHidden.value === "compare";
    statusText.textContent = isCompare
      ? "Analizando con YOLOv8n y YOLOv8s (puede tardar el doble)…"
      : "Analizando video (puede tardar varios minutos)…";

    const formData = new FormData(form);

    try {
      const res = await fetch("/upload", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Error desconocido");
      }

      if (data.mode === "compare") {
        window.location.href = `/compare/${data.job_id}`;
      } else {
        window.location.href = `/results/${data.job_id}`;
      }
    } catch (err) {
      progressSection.classList.add("d-none");
      errorSection.classList.remove("d-none");
      errorMsg.textContent = err.message;
      submitBtn.disabled = false;
    }
  });
});
