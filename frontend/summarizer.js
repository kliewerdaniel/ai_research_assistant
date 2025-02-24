// Import Transformers.js
import { pipeline } from '@xenova/transformers';

const BACKEND_URL = 'http://localhost:8000';

// Initialize the summarization pipeline
let summarizer = null;

// Lazy load the summarization model
async function loadSummarizer() {
    if (!summarizer) {
        try {
            summarizer = await pipeline('summarization', 'Xenova/distilbart-cnn-6-6');
        } catch (error) {
            console.error('Error loading summarizer:', error);
            throw error;
        }
    }
    return summarizer;
}

// Function to send article to backend
async function sendToBackend(title, originalContent, summary, metadata = {}) {
    try {
        const response = await fetch(`${BACKEND_URL}/add_article`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                title: title,
                content: originalContent,
                metadata: {
                    ...metadata,
                    summary: summary,
                    processed_at: new Date().toISOString()
                }
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(`Backend error: ${errorData.detail || response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Error sending to backend:', error);
        throw error;
    }
}

// Function to summarize text and send to backend
export async function summarizeText(text, title) {
    if (!text || !title) {
        throw new Error('Both text and title are required');
    }

    try {
        const model = await loadSummarizer();
        const result = await model(text, {
            max_length: 130,
            min_length: 30,
            do_sample: false
        });
        
        const summary = result[0].summary_text;
        
        // Send to backend
        const backendResponse = await sendToBackend(title, text, summary);
        console.log('Successfully saved to backend:', backendResponse);
        
        return {
            summary: summary,
            backendResponse: backendResponse
        };
    } catch (error) {
        console.error('Error during summarization or backend storage:', error);
        throw error;
    }
}
