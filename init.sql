CREATE TABLE IF NOT EXISTS wines(
    id INT PRIMARY KEY,
    wine_slug VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL
);

INSERT INTO wines(id, wine_slug, name) VALUES
(0, 'pobeda', 'побдеа'),
(1, 'kaberne-sovinon-2', 'каберне'),
(2, 'rubin-golodrigi', 'рубин голодриги')
ON CONFLICT DO NOTHING;