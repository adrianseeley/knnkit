const fs = require('fs');

const file = fs.readFileSync('log.csv', 'utf8');
const lines = file.split('\n');

const results = [];
for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(',');
    if (parts.length < 3) {
        continue;
    }
    const lastPart = parts[parts.length - 1];
    const fitness = parseFloat(lastPart);
    results.push({ fitness, line: lines[i] });
}

// sort low to high fitness
results.sort((a, b) => a.fitness - b.fitness);

const output = [];
output.push(lines[0]);
for (let i = 0; i < results.length; i++) {
    output.push(results[i].line);
}
fs.writeFileSync('log-sorted.csv', output.join('\n'));