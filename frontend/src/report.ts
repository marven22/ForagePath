import { jsPDF } from "jspdf";
import type { RecommendationResponse } from "./types";

const margin = 18;

function addWrapped(doc: jsPDF, text: string, x: number, y: number, width: number, lineHeight = 5): number {
  const lines = doc.splitTextToSize(text, width) as string[];
  doc.text(lines, x, y);
  return y + lines.length * lineHeight;
}

export function downloadRecommendationPdf(result: RecommendationResponse): void {
  const doc = new jsPDF({ unit: "mm", format: "letter" });
  const pageWidth = doc.internal.pageSize.getWidth();
  let y = 20;
  doc.setFillColor(23, 63, 45); doc.rect(0, 0, pageWidth, 34, "F");
  doc.setTextColor(255, 255, 255); doc.setFont("helvetica", "bold"); doc.setFontSize(20);
  doc.text("ForagePath recommendation", margin, 16);
  doc.setFont("helvetica", "normal"); doc.setFontSize(9);
  doc.text(`${result.request.state_code} · ${result.request.planting_class}`, margin, 24);
  y = 45; doc.setTextColor(23, 49, 38); doc.setFontSize(14); doc.setFont("helvetica", "bold");
  doc.text("Potential-forage mixture", margin, y); y += 8;
  doc.setFontSize(9); doc.setFont("helvetica", "normal");
  doc.text(`Spatial level: ${result.spatial_decision.label}`, margin, y);
  doc.text(`Recommendation source: ${result.spatial_decision.level === "statewide" ? "statewide consensus" : `${result.request.distance_to_modeled_cell_km.toFixed(1)} km away`}`, 105, y); y += 6;
  y = addWrapped(doc, result.spatial_decision.reason, margin, y, pageWidth - 2 * margin); y += 4;
  doc.setFont("helvetica", "bold"); doc.text("Selected-location evidence", margin, y); y += 5;
  doc.setFont("helvetica", "normal");
  const evidence = result.local_evidence;
  const shown = (value: number | null) => value == null ? "not available" : value.toLocaleString();
  y = addWrapped(doc, `Plant occurrence records: ${shown(evidence.plant_occurrence_records)} | Flowering observations: ${shown(evidence.flowering_observations)} | Bee occurrence records: ${shown(evidence.bee_occurrence_records)} | Bee sampling events: ${shown(evidence.bee_sampling_events)}.`, margin, y, pageWidth - 2 * margin); y += 4;
  result.recommendation.plants.forEach((plant, index) => {
    if (y > 235) { doc.addPage(); y = 20; }
    doc.setFillColor(240, 244, 235); doc.roundedRect(margin, y, pageWidth - 2 * margin, 28, 2, 2, "F");
    doc.setFont("helvetica", "bold"); doc.setFontSize(11);
    doc.text(`${index + 1}. ${plant.common_name || plant.scientific_name}`, margin + 4, y + 7);
    doc.setFont("helvetica", "italic"); doc.setFontSize(8); doc.text(plant.scientific_name, margin + 4, y + 13);
    doc.setFont("helvetica", "normal");
    const detail = [plant.growth_form, plant.moisture_guidance, plant.light_guidance].filter(Boolean).join(" · ");
    doc.text(doc.splitTextToSize(detail, pageWidth - 2 * margin - 8), margin + 4, y + 19);
    doc.text(`Relative suitability: ${plant.local_suitability_score?.toFixed(2) ?? "NA"} / 1 · Documented native bees: ${plant.documented_eligible_native_bee_species_count}`, margin + 4, y + 25);
    y += 33;
  });
  if (y > 220) { doc.addPage(); y = 20; }
  doc.setFont("helvetica", "bold"); doc.setFontSize(12); doc.text("Documented native-bee support", margin, y); y += 7;
  doc.setFont("helvetica", "normal"); doc.setFontSize(8);
  result.documented_native_bee_support.forEach((group) => {
    doc.text(`${group.bee_group.replaceAll("_", " ")}: ${group.species_count} species`, margin, y); y += 5;
  });
  y += 5; doc.setDrawColor(220, 210, 184); doc.line(margin, y, pageWidth - margin, y); y += 8;
  doc.setFont("helvetica", "bold"); doc.text("Scientific boundary", margin, y); y += 5;
  doc.setFont("helvetica", "normal");
  y = addWrapped(doc, result.interpretation.claim_boundary, margin, y, pageWidth - 2 * margin);
  y = addWrapped(doc, result.interpretation.important_note, margin, y + 2, pageWidth - 2 * margin);
  doc.setFontSize(7); doc.setTextColor(90, 100, 94);
  doc.text("Site guidance is a coarse screen. Confirm spacing, preparation, stock and establishment practices locally.", margin, y + 5);
  const place = (result.spatial_decision.county_name || result.spatial_decision.level).replaceAll(" ", "-");
  doc.save(`ForagePath-${result.request.state_code}-${place}.pdf`);
}
