-- Migration: Add conversation_id to document_chunks
-- Description: Links document chunks to specific conversations for multi-user isolation

-- Add conversation_id column to document_chunks
ALTER TABLE document_chunks 
ADD COLUMN IF NOT EXISTS conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE;

-- Create index for faster queries filtered by conversation
CREATE INDEX IF NOT EXISTS document_chunks_conversation_id_idx ON document_chunks(conversation_id);

-- Note: We can't create a composite btree+vector index directly.
-- The existing vector index (document_chunks_embedding_idx) will work with WHERE conversation_id = X
-- PostgreSQL will use both indexes efficiently (bitmap index scan).

-- Note: Existing chunks will have NULL conversation_id (orphaned chunks)
-- These can be cleaned up or left for backward compatibility

