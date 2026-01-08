-- Migration: Delete all data (Nuke/Reset)
-- WARNING: This will delete ALL data but keeps the tables! Use only for development/testing.
-- Description: Truncates all tables to remove all data while keeping table structure

-- Truncate tables in correct order (respecting foreign key constraints)
TRUNCATE TABLE messages CASCADE;
TRUNCATE TABLE document_chunks CASCADE;
TRUNCATE TABLE conversations CASCADE;

