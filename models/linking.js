// Entity relationship identification using Transformers.js
import { pipeline } from '@xenova/transformers';

class LinkingModel {
    constructor() {
        this.pipeline = null;
        this.nerPipeline = null;
        this.initialized = false;
    }

    async initialize() {
        if (!this.initialized) {
            try {
                // Initialize NER pipeline for entity detection
                this.nerPipeline = await pipeline('token-classification', 'Xenova/bert-base-NER');
                
                // Initialize zero-shot pipeline for relationship classification
                this.pipeline = await pipeline('zero-shot-classification', 'Xenova/distilbert-base-uncased');
                
                this.initialized = true;
            } catch (error) {
                console.error('Error initializing linking model:', error);
                throw error;
            }
        }
    }

    async identifyEntities(text) {
        if (!this.initialized) {
            await this.initialize();
        }

        try {
            const entities = await this.nerPipeline(text);
            
            // Group together consecutive tokens of the same entity
            const mergedEntities = [];
            let currentEntity = null;

            for (const entity of entities) {
                if (currentEntity && 
                    currentEntity.entity_group === entity.entity_group && 
                    currentEntity.end === entity.start) {
                    // Extend current entity
                    currentEntity.word += entity.word.replace('##', '');
                    currentEntity.end = entity.end;
                    currentEntity.score = Math.min(currentEntity.score, entity.score);
                } else {
                    // Start new entity
                    if (currentEntity) {
                        mergedEntities.push(currentEntity);
                    }
                    currentEntity = {
                        word: entity.word,
                        entity_group: entity.entity_group,
                        start: entity.start,
                        end: entity.end,
                        score: entity.score
                    };
                }
            }
            
            if (currentEntity) {
                mergedEntities.push(currentEntity);
            }

            return mergedEntities;
        } catch (error) {
            console.error('Error identifying entities:', error);
            throw error;
        }
    }

    async findRelationships(text) {
        if (!this.initialized) {
            await this.initialize();
        }

        try {
            // First identify entities in the text
            const entities = await this.identifyEntities(text);
            const relationships = [];

            // Define relationship types to check for
            const relationshipTypes = [
                'works for',
                'located in',
                'part of',
                'associated with',
                'created by',
                'belongs to'
            ];

            // Check for relationships between pairs of entities
            for (let i = 0; i < entities.length; i++) {
                for (let j = i + 1; j < entities.length; j++) {
                    const entity1 = entities[i];
                    const entity2 = entities[j];

                    // Extract the text segment between and including the entities
                    const startIdx = Math.min(entity1.start, entity2.start);
                    const endIdx = Math.max(entity1.end, entity2.end);
                    const segment = text.slice(startIdx, endIdx);

                    // Classify the relationship
                    const result = await this.pipeline(segment, relationshipTypes);
                    
                    // Only include relationships with confidence above threshold
                    const threshold = 0.3;
                    const topRelationship = {
                        type: result.labels[0],
                        confidence: result.scores[0]
                    };

                    if (topRelationship.confidence > threshold) {
                        relationships.push({
                            entity1: {
                                text: entity1.word,
                                type: entity1.entity_group
                            },
                            entity2: {
                                text: entity2.word,
                                type: entity2.entity_group
                            },
                            relationship: topRelationship
                        });
                    }
                }
            }

            return {
                entities,
                relationships,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            console.error('Error finding relationships:', error);
            throw error;
        }
    }
}

export default LinkingModel;
