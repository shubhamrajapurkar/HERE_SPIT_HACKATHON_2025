window.addEventListener('DOMContentLoaded', () => {
  const params = new URLSearchParams(window.location.search);
  const modelUrl = params.get('model');
  if (modelUrl) {
    document.getElementById('model').setAttribute('src', modelUrl);
  } else {
    alert('No model specified!');
  }
});
