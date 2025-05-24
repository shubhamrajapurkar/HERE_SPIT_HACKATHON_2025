// --- CONFIGURATION ---
const BLAND_AI_AUTHORIZATION_KEY = 'org_f9cec5629aa9cfcb5ca0412c28a7e1dddf67b78336f1f1d74ea4cf27534962998ad43806955df48a3a1669';
const TARGET_PHONE_NUMBER = '+919766281434';
const USER_PATHWAY_ID = '70101449-8e63-4b47-9b3d-c703b79be61a';
const GEMINI_API_KEY = "AIzaSyBGzJ6Jt45kBjbVacPHaw6MVbyS6GvDvEg"; // Ensure this is your actual, working key

if (GEMINI_API_KEY === "YOUR_GEMINI_API_KEY") {
    alert("Please replace 'YOUR_GEMINI_API_KEY' in app.js with your actual Gemini API key.");
}

const blandApiBaseUrl = 'https://api.bland.ai/v1';

const statusDiv = document.getElementById('status');
const automatedBookingButton = document.getElementById('automatedBookingButton');

// Add dummy "Send Confirmation to Mail" button logic
const sendConfirmationBtn = document.getElementById('sendConfirmationBtn');
if (sendConfirmationBtn) {
    sendConfirmationBtn.onclick = function() {
        alert('Sending confirmation to kushnhayade14@gmail.com');
    };
}

// --- HELPER FUNCTIONS ---
function updateStatus(message, isError = false) {
    if (!statusDiv) return;
    statusDiv.style.display = "block";
    statusDiv.innerHTML = ""; // Only show the latest message
    const p = document.createElement('p');
    p.textContent = message;
    if (isError) {
        p.style.color = 'red';
    }
    statusDiv.appendChild(p);
    statusDiv.scrollTop = statusDiv.scrollHeight;
}

function downloadTextFile(filename, text) {
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
    element.setAttribute('download', filename);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
    // updateStatus(`Summary downloaded as ${filename}`);
}

// --- MAIN LOGIC ---
if (automatedBookingButton) {
    automatedBookingButton.addEventListener('click', async () => {
        if (GEMINI_API_KEY === "YOUR_GEMINI_API_KEY" || !GEMINI_API_KEY) {
            // updateStatus("Error: Gemini API Key is not set or is a placeholder. Please update call.js.", true);
            return;
        }

        automatedBookingButton.disabled = true;
        statusDiv.style.display = "none"; // Hide status box before starting

        try {
            // updateStatus('Initiating automated booking call for Sahyadri Hotel...');
            // Updated call request with Sahyadri booking details
            const callRequestBody = {
                phone_number: TARGET_PHONE_NUMBER,
                task: "Automated Booking at Sahyadri Hotel",
                pathway_id: USER_PATHWAY_ID,
                voice: "maya",
                first_sentence: "Hello, this is an automated booking call for Sahyadri Hotel. Please negotiate firmly and secure the booking immediately.",
                wait_for_greeting: true,
                block_interruptions: true,
                record: true,
            };

            // updateStatus(`Sending to Bland AI: ${JSON.stringify(callRequestBody, null, 2)}`);

            const callOptions = {
                method: 'POST',
                headers: {
                    'authorization': BLAND_AI_AUTHORIZATION_KEY,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(callRequestBody)
            };

            const callResponse = await fetch(`${blandApiBaseUrl}/calls`, callOptions);
            const responseText = await callResponse.text();

            if (!callResponse.ok) {
                let errorData;
                try {
                    errorData = JSON.parse(responseText);
                } catch (e) {
                    errorData = { message: "Unknown error", details: responseText };
                }
                throw new Error(`Bland AI booking call failed: ${callResponse.status} ${callResponse.statusText}. Details: ${JSON.stringify(errorData)}`);
            }
            let callData;
            try {
                callData = JSON.parse(responseText);
            } catch (e) {
                throw new Error(`Bland AI call initiation succeeded but response was not valid JSON: ${responseText}`);
            }
            // updateStatus(`Call initiated. Status: ${callData.status}, Call ID: ${callData.call_id}`);

            if (!callData.call_id) {
                throw new Error('Call ID not found in Bland AI response.');
            }
            const callId = callData.call_id;

            // updateStatus('Waiting a few seconds before fetching booking confirmation data...');
            await new Promise(resolve => setTimeout(resolve, 100000));

            // updateStatus(`Fetching booking confirmation data for call ID: ${callId}...`);
            const correctOptions = {
                method: 'GET',
                headers: {
                    'authorization': BLAND_AI_AUTHORIZATION_KEY
                }
            };

            const correctResponse = await fetch(`https://cors-anywhere.herokuapp.com/${blandApiBaseUrl}/calls/${callId}/correct`, correctOptions);
            if (!correctResponse.ok) {
                const errorData = await correctResponse.json().catch(() => ({ message: "Unknown error" }));
                throw new Error(`Bland AI booking data fetch failed: ${correctResponse.status} ${correctResponse.statusText}. Details: ${JSON.stringify(errorData)}`);
            }
            const correctionData = await correctResponse.json();
            // updateStatus('Booking confirmation data received:');
            // updateStatus(JSON.stringify(correctionData, null, 2));

            // updateStatus('Sending data to Gemini for final booking confirmation...');
            // Use GoogleGenerativeAI from global if using CDN, or keep as is if using import
            const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
            const model = genAI.getGenerativeModel({ model: "gemini-1.5-pro" });

            let contentToSummarize = JSON.stringify(correctionData);
            if (correctionData.transcripts && Array.isArray(correctionData.transcripts) && correctionData.transcripts.length > 0) {
                contentToSummarize = correctionData.transcripts.map(t => `${t.speaker}: ${t.text}`).join("\n");
            } else if (correctionData.corrected_transcript) {
                contentToSummarize = correctionData.corrected_transcript;
            }

            const prompt = `Please summarize the following booking call data for Sahyadri Hotel. Emphasize firm negotiation and ensure that the booking is confirmed:\n\n${contentToSummarize}`;
            const result = await model.generateContent(prompt);
            const geminiResponse = result.response;
            const summary = geminiResponse.text();

            // updateStatus('Final booking confirmation received from Gemini:');
            updateStatus(summary);

            downloadTextFile(`booking_confirmation_${callId}.txt`, summary);
            // updateStatus('Automated booking process completed! An email confirmation has been sent.');
        } catch (error) {
            // updateStatus(`An error occurred: ${error.message}`, true);
            // console.error("Detailed error:", error);
        } finally {
            automatedBookingButton.disabled = false;
        }
    });
}

// Add global function for automated booking so it can be triggered from the main page
window.initiateAutomatedBooking = async function() {
    try {
        // updateStatus('Initiating automated booking call for Sahyadri Hotel...');
        const callRequestBody = {
            phone_number: TARGET_PHONE_NUMBER,
            task: "Automated Booking at Sahyadri Hotel",
            pathway_id: USER_PATHWAY_ID,
            voice: "maya",
            first_sentence: "Hello, this is an automated booking call for Sahyadri Hotel. Please negotiate firmly and confirm the booking immediately.",
            wait_for_greeting: true,
            block_interruptions: true,
            record: true,
        };
        // updateStatus(`Sending to Bland AI: ${JSON.stringify(callRequestBody, null, 2)}`);
        const callOptions = {
            method: 'POST',
            headers: {
                'authorization': BLAND_AI_AUTHORIZATION_KEY,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(callRequestBody)
        };
        const callResponse = await fetch(`${blandApiBaseUrl}/calls`, callOptions);
        const responseText = await callResponse.text();
        if (!callResponse.ok) {
            let errorData;
            try { errorData = JSON.parse(responseText); } catch (e) { errorData = { message: "Unknown error", details: responseText }; }
            throw new Error(`Bland AI booking call failed: ${callResponse.status} ${callResponse.statusText}. Details: ${JSON.stringify(errorData)}`);
        }
        let callData;
        try {
            callData = JSON.parse(responseText);
        } catch (e) {
            throw new Error(`Bland AI call initiation succeeded but response was not valid JSON: ${responseText}`);
        }
        // updateStatus(`Call initiated. Status: ${callData.status}, Call ID: ${callData.call_id}`);
        if (!callData.call_id) {
            throw new Error('Call ID not found in Bland AI response.');
        }
        const callId = callData.call_id;
        // updateStatus('Waiting before fetching booking confirmation data...');
        await new Promise(resolve => setTimeout(resolve, 100000));
        // updateStatus(`Fetching booking confirmation data for call ID: ${callId}...`);
        const correctOptions = {
            method: 'GET',
            headers: { 'authorization': BLAND_AI_AUTHORIZATION_KEY }
        };
        const correctResponse = await fetch(`https://cors-anywhere.herokuapp.com/${blandApiBaseUrl}/calls/${callId}/correct`, correctOptions);
        if (!correctResponse.ok) {
            const errorData = await correctResponse.json().catch(() => ({ message: "Unknown error" }));
            throw new Error(`Bland AI booking data fetch failed: ${correctResponse.status} ${correctResponse.statusText}. Details: ${JSON.stringify(errorData)}`);
        }
        const correctionData = await correctResponse.json();
        // updateStatus('Booking confirmation data received:');
        // updateStatus(JSON.stringify(correctionData, null, 2));
        // updateStatus('Automated booking process completed successfully!');
    } catch (error) {
        // updateStatus(`An error occurred: ${error.message}`, true);
        // console.error("Detailed error:", error);
    }
};