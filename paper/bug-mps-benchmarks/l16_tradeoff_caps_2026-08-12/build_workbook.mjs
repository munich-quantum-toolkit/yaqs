import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const here = path.dirname(fileURLToPath(import.meta.url));
const outputDir = path.join(here, "..", "outputs", "bug-l16-tradeoff");
const outputPath = path.join(outputDir, "bug_l16_benchmark_data.xlsx");
const previewDir = path.join(outputDir, "previews");

const raw = JSON.parse(await fs.readFile(path.join(here, "raw_results.json"), "utf8"));
const validationText = await fs.readFile(path.join(here, "VALIDATION.md"), "utf8");

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");

async function importCsv(fileName, sheetName) {
  const csv = await fs.readFile(path.join(here, fileName), "utf8");
  const imported = await Workbook.fromCSV(csv, { sheetName });
  const source = imported.worksheets.getItem(sheetName);
  const values = source.getUsedRange(true).values;
  const target = workbook.worksheets.add(sheetName);
  target.getRangeByIndexes(0, 0, values.length, values[0].length).values = values;
  return target;
}

const tradeoff = await importCsv("tradeoff_all_points.csv", "Tradeoff");
const pareto = await importCsv("tradeoff_pareto_points.csv", "Pareto");
const caps = await importCsv("cap_study.csv", "Cap Study");
const timings = await importCsv("timing_samples.csv", "Timing Samples");
const checkpoints = await importCsv("bug_first_step_checkpoints.csv", "BUG Checkpoints");
const protocol = workbook.worksheets.add("Protocol");
const validation = workbook.worksheets.add("Validation");

const navy = "#17365D";
const blue = "#DCE6F1";
const paleBlue = "#EDF4FA";
const paleGreen = "#E2F0D9";
const paleOrange = "#FCE4D6";
const gray = "#666666";
const lightBorder = "#C9D3DF";

function styleHeader(range) {
  range.format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
    wrapText: true,
    borders: { bottom: { style: "medium", color: navy } },
  };
  range.format.rowHeight = 30;
}

function styleRawSheet(sheet, lastColumn, lastRow, tableName) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const full = sheet.getRange(`A1:${lastColumn}${lastRow}`);
  full.format.font = { name: "Aptos", size: 9 };
  full.format.verticalAlignment = "center";
  styleHeader(sheet.getRange(`A1:${lastColumn}1`));
  const table = sheet.tables.add(`A1:${lastColumn}${lastRow}`, true, tableName);
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  table.showFilterButton = true;
}

styleRawSheet(tradeoff, "AB", 55, "TradeoffTable");
tradeoff.getRange("A:A").format.columnWidth = 42;
tradeoff.getRange("B:C").format.columnWidth = 11;
tradeoff.getRange("D:H").format.columnWidth = 12;
tradeoff.getRange("I:N").format.columnWidth = 15;
tradeoff.getRange("O:T").format.columnWidth = 16;
tradeoff.getRange("U:X").format.columnWidth = 13;
tradeoff.getRange("Y:AB").format.columnWidth = 27;
tradeoff.getRange("D2:D55").format.numberFormat = "0.0000";
tradeoff.getRange("F2:F55").format.numberFormat = "0.0E+00";
tradeoff.getRange("J2:N55").format.numberFormat = "0.000";
tradeoff.getRange("O2:T55").format.numberFormat = "0.000E+00";
tradeoff.getRange("U2:W55").format.numberFormat = "0";

styleRawSheet(pareto, "AB", 22, "ParetoTable");
pareto.getRange("A:A").format.columnWidth = 42;
pareto.getRange("B:C").format.columnWidth = 11;
pareto.getRange("D:H").format.columnWidth = 12;
pareto.getRange("I:N").format.columnWidth = 15;
pareto.getRange("O:T").format.columnWidth = 16;
pareto.getRange("U:X").format.columnWidth = 13;
pareto.getRange("Y:AB").format.columnWidth = 27;
pareto.getRange("D2:D22").format.numberFormat = "0.0000";
pareto.getRange("F2:F22").format.numberFormat = "0.0E+00";
pareto.getRange("J2:N22").format.numberFormat = "0.000";
pareto.getRange("O2:T22").format.numberFormat = "0.000E+00";

styleRawSheet(caps, "T", 9, "CapStudyTable");
caps.getRange("A:A").format.columnWidth = 42;
caps.getRange("B:C").format.columnWidth = 11;
caps.getRange("D:F").format.columnWidth = 13;
caps.getRange("G:J").format.columnWidth = 16;
caps.getRange("K:O").format.columnWidth = 17;
caps.getRange("P:T").format.columnWidth = 15;
caps.getRange("D2:D9").format.numberFormat = "0.0000";
caps.getRange("E2:E9").format.numberFormat = "0.0E+00";
caps.getRange("H2:J9").format.numberFormat = "0.000";
caps.getRange("K2:O9").format.numberFormat = "0.000E+00";

const timingRows = (await fs.readFile(path.join(here, "timing_samples.csv"), "utf8")).trim().split(/\r?\n/).length;
styleRawSheet(timings, "K", timingRows, "TimingSamplesTable");
timings.getRange("A:A").format.columnWidth = 42;
timings.getRange("B:C").format.columnWidth = 11;
timings.getRange("D:G").format.columnWidth = 14;
timings.getRange("H:H").format.columnWidth = 17;
timings.getRange("I:K").format.columnWidth = 23;
timings.getRange(`D2:D${timingRows}`).format.numberFormat = "0.0000";
timings.getRange(`E2:E${timingRows}`).format.numberFormat = "0.0E+00";
timings.getRange(`H2:H${timingRows}`).format.numberFormat = "0.000";

const checkpointRows = (await fs.readFile(path.join(here, "bug_first_step_checkpoints.csv"), "utf8")).trim().split(/\r?\n/).length;
styleRawSheet(checkpoints, "I", checkpointRows, "BugCheckpointTable");
checkpoints.getRange("A:A").format.columnWidth = 42;
checkpoints.getRange("B:B").format.columnWidth = 11;
checkpoints.getRange("C:E").format.columnWidth = 14;
checkpoints.getRange("F:F").format.columnWidth = 23;
checkpoints.getRange("G:I").format.columnWidth = 15;
checkpoints.getRange(`C2:C${checkpointRows}`).format.numberFormat = "0.0000";
checkpoints.getRange(`D2:D${checkpointRows}`).format.numberFormat = "0.0E+00";

const protocolRows = [["Parameter", "Value"]];
for (const [key, value] of Object.entries(raw.protocol)) {
  protocolRows.push([key, typeof value === "object" ? JSON.stringify(value) : value]);
}
protocol.getRangeByIndexes(0, 0, protocolRows.length, 2).values = protocolRows;
protocol.showGridLines = false;
protocol.freezePanes.freezeRows(1);
styleHeader(protocol.getRange("A1:B1"));
protocol.getRange(`A1:B${protocolRows.length}`).format.font = { name: "Aptos", size: 10 };
protocol.getRange("A:A").format.columnWidth = 42;
protocol.getRange("B:B").format.columnWidth = 105;
protocol.getRange(`B2:B${protocolRows.length}`).format.wrapText = true;
protocol.tables.add(`A1:B${protocolRows.length}`, true, "ProtocolTable").style = "TableStyleMedium2";

const validationRows = [["Status", "Check", "Detail"]];
for (const line of validationText.split(/\r?\n/)) {
  const match = line.match(/^- (PASS|FAIL): (.*) \((.*)\)$/);
  if (match) validationRows.push([match[1], match[2], match[3]]);
}
validation.getRangeByIndexes(0, 0, validationRows.length, 3).values = validationRows;
validation.showGridLines = false;
validation.freezePanes.freezeRows(1);
styleHeader(validation.getRange("A1:C1"));
validation.getRange(`A1:C${validationRows.length}`).format.font = { name: "Aptos", size: 9 };
validation.getRange("A:A").format.columnWidth = 11;
validation.getRange("B:B").format.columnWidth = 95;
validation.getRange("C:C").format.columnWidth = 62;
validation.getRange(`B2:C${validationRows.length}`).format.wrapText = true;
validation.tables.add(`A1:C${validationRows.length}`, true, "ValidationTable").style = "TableStyleMedium2";
validation.getRange(`A2:A${validationRows.length}`).conditionalFormats.add("containsText", {
  text: "PASS",
  format: { fill: paleGreen, font: { color: "#375623" } },
});
validation.getRange(`A2:A${validationRows.length}`).conditionalFormats.add("containsText", {
  text: "FAIL",
  format: { fill: paleOrange, font: { color: "#9C0006", bold: true } },
});

summary.showGridLines = false;
summary.getRange("A1:I1").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
summary.getRange("A1:I1").merge();
summary.getRange("A1").values = [["L=16 BUG vs 2TDVP benchmark"]];
summary.getRange("A1:I1").format.rowHeight = 34;
summary.getRange("A2:I2").merge();
summary.getRange("A2").values = [["Validated runtime-accuracy sweep and active-cap study; raw timings and state diagnostics are retained on separate sheets."]];
summary.getRange("A2:I2").format = { font: { italic: true, color: gray }, wrapText: true };
summary.getRange("A2:I2").format.rowHeight = 30;

summary.getRange("A4:A6").values = [["Matched epsilon"], ["Matched chi cap"], ["Validation checks"]];
summary.getRange("B4:B5").values = [[1e-12], [512]];
summary.getRange("B6").formulas = [[`=COUNTIF('Validation'!$A$2:$A$${validationRows.length},"PASS")&" / "&COUNTA('Validation'!$A$2:$A$${validationRows.length})`]];
summary.getRange("A4:B6").format = {
  fill: paleBlue,
  borders: { preset: "outside", style: "thin", color: lightBorder },
};
summary.getRange("A4:A6").format.font = { bold: true };
summary.getRange("B4").format.numberFormat = "0.0E+00";
summary.getRange("B5").format.numberFormat = "0";

summary.getRange("A8:I8").values = [[
  "Model", "dt", "BUG time (s)", "2TDVP time (s)", "2TDVP/BUG", "BUG infidelity", "2TDVP infidelity", "BUG max chi", "2TDVP max chi",
]];
styleHeader(summary.getRange("A8:I8"));
const matchedRows = [];
for (const model of ["tfim", "hs"]) {
  for (const dt of [0.01, 0.005, 0.0025]) matchedRows.push([model, dt]);
}
summary.getRange("A9:B14").values = matchedRows;
for (let row = 9; row <= 14; row += 1) {
  const criteria = `'Tradeoff'!$B$2:$B$55,$A${row},'Tradeoff'!$D$2:$D$55,$B${row},'Tradeoff'!$F$2:$F$55,$B$4,'Tradeoff'!$G$2:$G$55,$B$5`;
  summary.getRange(`C${row}`).formulas = [[`=SUMIFS('Tradeoff'!$J$2:$J$55,${criteria},'Tradeoff'!$C$2:$C$55,"bug")`]];
  summary.getRange(`D${row}`).formulas = [[`=SUMIFS('Tradeoff'!$J$2:$J$55,${criteria},'Tradeoff'!$C$2:$C$55,"2tdvp")`]];
  summary.getRange(`E${row}`).formulas = [[`=D${row}/C${row}`]];
  summary.getRange(`F${row}`).formulas = [[`=SUMIFS('Tradeoff'!$P$2:$P$55,${criteria},'Tradeoff'!$C$2:$C$55,"bug")`]];
  summary.getRange(`G${row}`).formulas = [[`=SUMIFS('Tradeoff'!$P$2:$P$55,${criteria},'Tradeoff'!$C$2:$C$55,"2tdvp")`]];
  summary.getRange(`H${row}`).formulas = [[`=SUMIFS('Tradeoff'!$U$2:$U$55,${criteria},'Tradeoff'!$C$2:$C$55,"bug")`]];
  summary.getRange(`I${row}`).formulas = [[`=SUMIFS('Tradeoff'!$U$2:$U$55,${criteria},'Tradeoff'!$C$2:$C$55,"2tdvp")`]];
}
summary.getRange("A9:I14").format.borders = { insideHorizontal: { style: "thin", color: lightBorder } };
summary.getRange("B9:B14").format.numberFormat = "0.0000";
summary.getRange("C9:D14").format.numberFormat = "0.000";
summary.getRange("E9:E14").format.numberFormat = "0.000";
summary.getRange("F9:G14").format.numberFormat = "0.000E+00";
summary.getRange("H9:I14").format.numberFormat = "0";

summary.getRange("A17:H17").values = [[
  "chi cap", "BUG time (s)", "2TDVP time (s)", "2TDVP/BUG", "BUG infidelity", "2TDVP infidelity", "BUG attained chi", "2TDVP attained chi",
]];
styleHeader(summary.getRange("A17:H17"));
summary.getRange("A18:A21").values = [[32], [64], [96], [512]];
for (let row = 18; row <= 21; row += 1) {
  const criteria = `'Cap Study'!$F$2:$F$9,$A${row}`;
  summary.getRange(`B${row}`).formulas = [[`=SUMIFS('Cap Study'!$H$2:$H$9,${criteria},'Cap Study'!$C$2:$C$9,"bug")`]];
  summary.getRange(`C${row}`).formulas = [[`=SUMIFS('Cap Study'!$H$2:$H$9,${criteria},'Cap Study'!$C$2:$C$9,"2tdvp")`]];
  summary.getRange(`D${row}`).formulas = [[`=C${row}/B${row}`]];
  summary.getRange(`E${row}`).formulas = [[`=SUMIFS('Cap Study'!$L$2:$L$9,${criteria},'Cap Study'!$C$2:$C$9,"bug")`]];
  summary.getRange(`F${row}`).formulas = [[`=SUMIFS('Cap Study'!$L$2:$L$9,${criteria},'Cap Study'!$C$2:$C$9,"2tdvp")`]];
  summary.getRange(`G${row}`).formulas = [[`=SUMIFS('Cap Study'!$P$2:$P$9,${criteria},'Cap Study'!$C$2:$C$9,"bug")`]];
  summary.getRange(`H${row}`).formulas = [[`=SUMIFS('Cap Study'!$P$2:$P$9,${criteria},'Cap Study'!$C$2:$C$9,"2tdvp")`]];
}
summary.getRange("A18:H21").format.borders = { insideHorizontal: { style: "thin", color: lightBorder } };
summary.getRange("A18:A21").format.numberFormat = "0";
summary.getRange("B18:D21").format.numberFormat = "0.000";
summary.getRange("E18:F21").format.numberFormat = "0.000E+00";
summary.getRange("G18:H21").format.numberFormat = "0";

summary.getRange("A24:I26").values = [[
  "Reading guide", "Open markers in the PDF/PNG figure are dominated grid points; connected filled markers are the final Pareto envelopes.", null, null, null, null, null, null, null,
], [
  "Timing policy", "Medians use three samples for all Pareto and cap points; construction, exact references, diagnostics, warm-up, and file output are excluded.", null, null, null, null, null, null, null,
], [
  "Accuracy caveat", "Matched-parameter accuracy is model- and regime-dependent. Use the runtime-accuracy envelope rather than asserting a universal accuracy advantage.", null, null, null, null, null, null, null,
]];
for (let row = 24; row <= 26; row += 1) summary.getRange(`B${row}:I${row}`).merge();
summary.getRange("A24:A26").format = { fill: blue, font: { bold: true } };
summary.getRange("B24:I26").format = { wrapText: true, font: { color: gray } };
summary.getRange("A24:I26").format.borders = { preset: "outside", style: "thin", color: lightBorder };
summary.getRange("A24:I26").format.rowHeight = 34;

summary.getRange("A:I").format.font = { name: "Aptos", size: 10 };
summary.getRange("A:A").format.columnWidth = 21;
summary.getRange("B:B").format.columnWidth = 14;
summary.getRange("C:D").format.columnWidth = 17;
summary.getRange("E:E").format.columnWidth = 14;
summary.getRange("F:G").format.columnWidth = 19;
summary.getRange("H:I").format.columnWidth = 17;
summary.freezePanes.freezeRows(2);

await fs.mkdir(previewDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
for (const [sheetName, fileName, range] of [
  ["Summary", "summary.png", "A1:I26"],
  ["Tradeoff", "tradeoff.png", "A1:L16"],
  ["Pareto", "pareto.png", "A1:L16"],
  ["Cap Study", "cap-study.png", "A1:T9"],
  ["Timing Samples", "timing-samples.png", "A1:K16"],
  ["BUG Checkpoints", "bug-checkpoints.png", "A1:I16"],
  ["Protocol", "protocol.png", `A1:B${protocolRows.length}`],
  ["Validation", "validation.png", "A1:C18"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, fileName), new Uint8Array(await preview.arrayBuffer()));
}

const inspect = await workbook.inspect({
  kind: "table",
  sheetId: "Summary",
  range: "A1:I26",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 10,
});
await fs.writeFile(path.join(outputDir, "summary_inspection.ndjson"), inspect.ndjson, "utf8");
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(path.join(outputDir, "formula_error_scan.ndjson"), errors.ndjson, "utf8");

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);
