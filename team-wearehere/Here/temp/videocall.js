// Hide the watermark dynamically using MutationObserver
function observeAndHideWatermark() {
    const observer = new MutationObserver(() => {
        const watermark = document.querySelector('.watermark'); // Select the watermark element
        const leftWatermark = document.querySelector('.leftwatermark'); // Select the left watermark element
        if (watermark) watermark.style.display = 'none'; // Hide the watermark
        if (leftWatermark) leftWatermark.style.display = 'none'; // Hide the left watermark
    });

    // Observe changes in the video container
    const videoContainer = document.getElementById('video-container');
    if (videoContainer) {
        observer.observe(videoContainer, { childList: true, subtree: true });
    }
}

// Initialize the video call
function initializeVideoCall() {
    const domain = 'meet.jit.si'; // Jitsi Meet server
    const options = {
        roomName: `MyCustomRoom-${Date.now()}`, // Generate a unique room name
        parentNode: document.getElementById('video-container'),
        width: '100%',
        height: '100%',
        configOverwrite: {
            startWithAudioMuted: true,
            startWithVideoMuted: false,
        },
        interfaceConfigOverwrite: {
            SHOW_JITSI_WATERMARK: false, // Disable Jitsi watermark
            SHOW_BRAND_WATERMARK: false, // Disable brand watermark
            SHOW_POWERED_BY: false, // Disable "Powered by Jitsi" text
            DEFAULT_REMOTE_DISPLAY_NAME: 'Guest',
        },
    };

    // Create the Jitsi Meet API object
    const api = new JitsiMeetExternalAPI(domain, options);

    // Observe and hide the watermark dynamically
    observeAndHideWatermark();

    // Handle events (optional)
    api.addEventListener('videoConferenceJoined', () => {
        console.log('Video conference joined');
    });

    api.addEventListener('videoConferenceLeft', () => {
        console.log('Video conference left');
    });
}

// Add a "Return to Map" button
function addReturnToMapButton() {
    const returnBtn = document.createElement('button');
    returnBtn.id = 'return-to-map-btn';
    returnBtn.textContent = 'Return to Map';
    returnBtn.className = 'bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-lg mt-4';
    returnBtn.style.position = 'absolute';
    returnBtn.style.top = '10px';
    returnBtn.style.right = '10px';
    returnBtn.style.zIndex = '1000';
    returnBtn.onclick = function () {
        window.location.href = 'http://127.0.0.1:5500/mapbox.html?src=Kalyan&dest=Pune'; // Redirect to map page
    };

    document.body.appendChild(returnBtn);
}

// Initialize the video call when the page loads
document.addEventListener('DOMContentLoaded', () => {
    initializeVideoCall();
    addReturnToMapButton();
});
