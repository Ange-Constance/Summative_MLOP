export const API_URL = "https://summative-mlop-4xmz.onrender.com";

export async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  return await response.json();
}
