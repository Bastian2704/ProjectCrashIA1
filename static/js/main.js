document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("upload-form");
  if (!form) return;

  const submitBtn = document.getElementById("submit-btn");
  const progressSection = document.getElementById("progress-section");
  const statusText = document.getElementById("status-text");
  const errorSection = document.getElementById("error-section");
  const errorMsg = document.getElementById("error-msg");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    errorSection.classList.add("d-none");
    progressSection.classList.remove("d-none");
    submitBtn.disabled = true;
    statusText.textContent = "Subiendo video…";

    const formData = new FormData(form);

    try {
      statusText.textContent = "Analizando video (puede tardar varios minutos)…";
      const res = await fetch("/upload", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Error desconocido");
      }

      window.location.href = `/results/${data.job_id}`;
    } catch (err) {
      progressSection.classList.add("d-none");
      errorSection.classList.remove("d-none");
      errorMsg.textContent = err.message;
      submitBtn.disabled = false;
    }
  });
});
