// Keyword and topic extraction using Transformers.js
import { pipeline } from '@xenova/transformers';

class TaggingModel {
    constructor() {
        this.pipeline = null;
        this.initialized = false;
    }

    async initialize() {
        if (!this.initialized) {
            try {
                // Initialize the zero-shot classification pipeline
                this.pipeline = await pipeline('zero-shot-classification', 'Xenova/distilbert-base-uncased');
                this.initialized = true;
            } catch (error) {
                console.error('Error initializing tagging model:', error);
                throw error;
            }
        }
    }

    async extractKeywords(text, candidateLabels = ['technology', 'science', 'business', 'health', 'politics']) {
        if (!this.initialized) {
            await this.initialize();
        }

        try {
            // Perform zero-shot classification
            const result = await this.pipeline(text, candidateLabels);
            
            // Filter results to only include labels with confidence above threshold
            const threshold = 0.3;
            const relevantTags = result.labels
                .filter((_, index) => result.scores[index] > threshold)
                .map((label, index) => ({
                    tag: label,
                    confidence: result.scores[index]
                }));

            return {
                tags: relevantTags,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            console.error('Error extracting keywords:', error);
            throw error;
        }
    }

    // Extract topics using text chunks
    async extractTopics(text) {
        // Split text into chunks if it's too long
        const chunks = this.splitIntoChunks(text);
        const topics = new Set();

        for (const chunk of chunks) {
            const result = await this.extractKeywords(chunk);
            result.tags.forEach(tag => topics.add(tag.tag));
        }

        return Array.from(topics);
    }

    // Helper method to split text into manageable chunks
    splitIntoChunks(text, maxLength = 512) {
        const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
        const chunks = [];
        let currentChunk = '';

        for (const sentence of sentences) {
            if ((currentChunk + sentence).length <= maxLength) {
                currentChunk += sentence;
            } else {
                if (currentChunk) chunks.push(currentChunk.trim());
                currentChunk = sentence;
            }
        }

        if (currentChunk) chunks.push(currentChunk.trim());
        return chunks;
    }
}

export default TaggingModel;
