-- database/schema.sql
--
-- Run this once against your MySQL server to create the HandTalk AI
-- database and its tables. This is completely separate from the AI
-- pipeline (TensorFlow models, training data, etc.) and only supports
-- the web dashboard's login system and sentence history.
--
-- Usage (from a terminal, with MySQL installed):
--   mysql -u root -p < database/schema.sql
-- Or paste the contents into MySQL Workbench / phpMyAdmin's SQL tab.

CREATE DATABASE IF NOT EXISTS handtalk_ai
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE handtalk_ai;

-- ── users ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    fullname        VARCHAR(150)    NOT NULL,
    email           VARCHAR(255)    NOT NULL UNIQUE,
    username        VARCHAR(50)     NOT NULL UNIQUE,
    password_hash   VARCHAR(255)    NOT NULL,
    avatar_path     VARCHAR(255)    NULL,
    last_active     TIMESTAMP       NULL,
    created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ── sessions ─────────────────────────────────────────────────────────────
-- One row per login. logout_time stays NULL while the session is active.
-- duration is a generated column - MySQL computes it automatically from
-- login_time/logout_time the moment logout_time is set. total_gestures
-- and total_sentences are reserved for the upcoming Analytics page.
-- Created before sentence_history below, since that table now references it.
CREATE TABLE IF NOT EXISTS sessions (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    user_id         INT             NOT NULL,
    login_time      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    logout_time     TIMESTAMP       NULL,
    duration        INT GENERATED ALWAYS AS
                        (TIMESTAMPDIFF(SECOND, login_time, logout_time)) STORED,
    total_gestures  INT             NOT NULL DEFAULT 0,
    total_sentences INT             NOT NULL DEFAULT 0,
    CONSTRAINT fk_sessions_user
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE,
    INDEX idx_sessions_user_id (user_id)
) ENGINE=InnoDB;

-- ── sentence_history ────────────────────────────────────────────────────
-- One row per sentence a logged-in user has spoken through the dashboard.
-- session_id links back to the login session it was spoken during (the
-- generated camera/socket session, not the PHP-style web session) and is
-- nullable since older rows or edge cases may not have one.
CREATE TABLE IF NOT EXISTS sentence_history (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    user_id         INT             NOT NULL,
    session_id      INT             NULL,
    sentence        VARCHAR(500)    NOT NULL,
    created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_sentence_history_user
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_sentence_history_session
        FOREIGN KEY (session_id) REFERENCES sessions(id)
        ON DELETE SET NULL,
    INDEX idx_sentence_history_user_id (user_id)
) ENGINE=InnoDB;

-- gesture_history (optional, deferred) will be added in a later phase
-- once we've decided exactly what gesture-level data is worth logging.

-- ── user_settings ────────────────────────────────────────────────────────
-- One row per user, holding their Settings-page preferences (AI/Speech,
-- Camera, Dashboard). UNIQUE on user_id (not user_id as the PK itself)
-- so settings_service.py can use a plain auto-increment id like every
-- other table here, while still enforcing "one settings row per user."
-- Defaults mirror the ones already specified on the Settings page, so a
-- fresh row inserted with no explicit values still behaves correctly.
CREATE TABLE IF NOT EXISTS user_settings (
    id                  INT AUTO_INCREMENT PRIMARY KEY,
    user_id             INT             NOT NULL,
    auto_speak          TINYINT(1)      NOT NULL DEFAULT 0,
    speech_speed        DECIMAL(3,2)    NOT NULL DEFAULT 1.00,
    speech_volume       TINYINT         NOT NULL DEFAULT 100,
    camera_size         ENUM('small','medium','large') NOT NULL DEFAULT 'medium',
    show_analytics      TINYINT(1)      NOT NULL DEFAULT 1,
    show_history_panel  TINYINT(1)      NOT NULL DEFAULT 1,
    created_at          TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP
                                        ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT fk_user_settings_user
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE,
    UNIQUE KEY uq_user_settings_user_id (user_id)
) ENGINE=InnoDB;

-- ── favorites ────────────────────────────────────────────────────────────
-- One row per sentence a user has starred from their History page. The
-- unique key on (user_id, sentence_id) means a sentence can only be
-- favorited once per user - favoriting it again is just a no-op rather
-- than a duplicate row, which keeps the star toggle on the frontend
-- simple (no need to check existence client-side first).
CREATE TABLE IF NOT EXISTS favorites (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    user_id         INT             NOT NULL,
    sentence_id     INT             NOT NULL,
    created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_favorites_user
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE,
    CONSTRAINT fk_favorites_sentence
        FOREIGN KEY (sentence_id) REFERENCES sentence_history(id)
        ON DELETE CASCADE,
    UNIQUE KEY uq_favorites_user_sentence (user_id, sentence_id),
    INDEX idx_favorites_user_id (user_id)
) ENGINE=InnoDB;

-- ── password_resets ──────────────────────────────────────────────────────
-- One row per "forgot password" request. reset_code is stored as a
-- bcrypt-style hash (via password_reset_service.hash_code), never in
-- plaintext, same principle as users.password_hash - so a leaked
-- database dump can't be used to replay anyone's reset code. expires_at
-- is set to created_at + 10 minutes at insert time; used_at is set once
-- a code has been successfully consumed, so it can never be reused even
-- if it hasn't expired yet (see password_reset_service.verify_reset_code).
CREATE TABLE IF NOT EXISTS password_resets (
    id              INT AUTO_INCREMENT PRIMARY KEY,
    user_id         INT             NOT NULL,
    email           VARCHAR(255)    NOT NULL,
    reset_code      VARCHAR(255)    NOT NULL,
    expires_at      TIMESTAMP       NOT NULL,
    used_at         TIMESTAMP       NULL,
    created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_password_resets_user
        FOREIGN KEY (user_id) REFERENCES users(id)
        ON DELETE CASCADE,
    INDEX idx_password_resets_user_id (user_id),
    INDEX idx_password_resets_email (email)
) ENGINE=InnoDB;

-- ── Migration for an existing database ──────────────────────────────────
-- Your tables already exist from earlier phases, so the CREATE TABLE
-- statements above won't re-run. Run this once in phpMyAdmin's SQL tab
-- instead, to add the new column to your existing sentence_history table:
--
-- ALTER TABLE sentence_history
--   ADD COLUMN session_id INT NULL AFTER user_id,
--   ADD CONSTRAINT fk_sentence_history_session
--       FOREIGN KEY (session_id) REFERENCES sessions(id)
--       ON DELETE SET NULL;
--
-- And to add the new favorites table itself (paste into phpMyAdmin's SQL
-- tab against the handtalk_ai database - safe to run even if it already
-- exists, thanks to IF NOT EXISTS):
--
-- CREATE TABLE IF NOT EXISTS favorites (
--     id              INT AUTO_INCREMENT PRIMARY KEY,
--     user_id         INT             NOT NULL,
--     sentence_id     INT             NOT NULL,
--     created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
--     CONSTRAINT fk_favorites_user
--         FOREIGN KEY (user_id) REFERENCES users(id)
--         ON DELETE CASCADE,
--     CONSTRAINT fk_favorites_sentence
--         FOREIGN KEY (sentence_id) REFERENCES sentence_history(id)
--         ON DELETE CASCADE,
--     UNIQUE KEY uq_favorites_user_sentence (user_id, sentence_id),
--     INDEX idx_favorites_user_id (user_id)
-- ) ENGINE=InnoDB;
--
-- And to add the two new Profile-feature columns to your existing users
-- table (paste into phpMyAdmin's SQL tab - safe to skip if you already
-- have them):
--
-- ALTER TABLE users
--   ADD COLUMN avatar_path VARCHAR(255) NULL,
--   ADD COLUMN last_active TIMESTAMP NULL;
--
-- And to add the new user_settings table for the Settings page (paste
-- into phpMyAdmin's SQL tab against the handtalk_ai database - safe to
-- run even if it already exists, thanks to IF NOT EXISTS):
--
-- CREATE TABLE IF NOT EXISTS user_settings (
--     id                  INT AUTO_INCREMENT PRIMARY KEY,
--     user_id             INT             NOT NULL,
--     auto_speak          TINYINT(1)      NOT NULL DEFAULT 0,
--     speech_speed        DECIMAL(3,2)    NOT NULL DEFAULT 1.00,
--     speech_volume       TINYINT         NOT NULL DEFAULT 100,
--     camera_size         ENUM('small','medium','large') NOT NULL DEFAULT 'medium',
--     show_analytics      TINYINT(1)      NOT NULL DEFAULT 1,
--     show_history_panel  TINYINT(1)      NOT NULL DEFAULT 1,
--     created_at          TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
--     updated_at          TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP
--                                         ON UPDATE CURRENT_TIMESTAMP,
--     CONSTRAINT fk_user_settings_user
--         FOREIGN KEY (user_id) REFERENCES users(id)
--         ON DELETE CASCADE,
--     UNIQUE KEY uq_user_settings_user_id (user_id)
-- ) ENGINE=InnoDB;
--
-- And to add the new password_resets table for the Forgot Password
-- feature (paste into phpMyAdmin's SQL tab against the handtalk_ai
-- database - safe to run even if it already exists, thanks to
-- IF NOT EXISTS):
--
-- CREATE TABLE IF NOT EXISTS password_resets (
--     id              INT AUTO_INCREMENT PRIMARY KEY,
--     user_id         INT             NOT NULL,
--     email           VARCHAR(255)    NOT NULL,
--     reset_code      VARCHAR(255)    NOT NULL,
--     expires_at      TIMESTAMP       NOT NULL,
--     used_at         TIMESTAMP       NULL,
--     created_at      TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,
--     CONSTRAINT fk_password_resets_user
--         FOREIGN KEY (user_id) REFERENCES users(id)
--         ON DELETE CASCADE,
--     INDEX idx_password_resets_user_id (user_id),
--     INDEX idx_password_resets_email (email)
-- ) ENGINE=InnoDB;
--
-- And to drop the show_confidence column from an existing user_settings
-- table (the "Show Confidence Percentage" toggle was removed from the
-- Settings page - safe to skip if your table doesn't have this column):
--
-- ALTER TABLE user_settings
--   DROP COLUMN show_confidence;