-- Database (optional; create ahead if you prefer to control grants)
-- CREATE DATABASE IF NOT EXISTS `avengers_cache` DEFAULT CHARACTER SET utf8mb4;

-- Table
CREATE TABLE IF NOT EXISTS `http_cache` (
  `cache_key` VARCHAR(128) PRIMARY KEY,
  `status` INT NOT NULL,
  `headers` JSON NOT NULL,
  `body` LONGBLOB NOT NULL,
  `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `ttl_sec` INT NULL,
  `expires_at` TIMESTAMP NULL,
  INDEX (`expires_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
