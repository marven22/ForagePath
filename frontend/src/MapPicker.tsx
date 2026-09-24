import { useEffect } from "react";
import type { GeoJsonObject } from "geojson";
import { CircleMarker, GeoJSON, MapContainer, Rectangle, TileLayer, Tooltip, useMap, useMapEvents } from "react-leaflet";
import type { BoundaryCollection, GridCell } from "./types";
import "leaflet/dist/leaflet.css";

function MapController({ center, zoom }: { center: [number, number]; zoom: number }) {
  const map = useMap();
  useEffect(() => { map.flyTo(center, zoom, { duration: .6 }); }, [center, zoom, map]);
  return null;
}

function ClickHandler({ onChoose }: { onChoose: (latitude: number, longitude: number) => void }) {
  useMapEvents({ click: (event) => onChoose(event.latlng.lat, event.latlng.lng) });
  return null;
}

export default function MapPicker({
  cells, boundaries, center, selected, onChoose,
}: {
  cells: GridCell[];
  boundaries: BoundaryCollection | null;
  center: [number, number];
  selected: [number, number];
  onChoose: (latitude: number, longitude: number) => void;
}) {
  const latitudes = cells.map((cell) => cell.center_latitude);
  const longitudes = cells.map((cell) => cell.center_longitude);
  const span = Math.max(
    Math.max(...latitudes) - Math.min(...latitudes),
    Math.max(...longitudes) - Math.min(...longitudes),
  );
  const zoom = span > 25 ? 3 : span > 15 ? 4 : span > 8 ? 5 : span > 4 ? 6 : 7;
  return (
    <div className="map-picker">
      <MapContainer center={center} zoom={7} scrollWheelZoom className="map-canvas">
        <TileLayer
          url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>'
        />
        <MapController center={center} zoom={zoom} />
        <ClickHandler onChoose={onChoose} />
        {boundaries && <GeoJSON
          key={String(boundaries.features[0]?.properties.GEOID)}
          data={boundaries as unknown as GeoJsonObject}
          style={(feature) => feature?.properties?.kind === "state"
            ? { color: "#183f32", weight: 2.5, fillOpacity: 0 }
            : { color: "#62736c", weight: .7, fillOpacity: 0 }}
          onEachFeature={(feature, layer) => {
            if (feature.properties?.kind === "county") layer.bindTooltip(`${feature.properties.BASENAME} County`);
          }}
        />}
        {cells.map((cell) => {
          const half = cell.resolution_degrees / 2;
          const appearance = cell.evidence_grade === "strong_local"
            ? { color: "#26734d", fillColor: "#43a36f", fillOpacity: .20 }
            : cell.evidence_grade === "modeled_limited"
              ? { color: "#a66b16", fillColor: "#e8b44b", fillOpacity: .16 }
              : { color: "#77817d", fillColor: "#aeb6b2", fillOpacity: .10 };
          const status = cell.evidence_grade === "strong_local" ? "validated local recommendation"
            : cell.evidence_grade === "modeled_limited" ? "replay-supported local guidance"
              : "modeled environmental area; no published local recommendation";
          return (
            <Rectangle
              key={cell.cell_id}
              bounds={[[cell.center_latitude - half, cell.center_longitude - half], [cell.center_latitude + half, cell.center_longitude + half]]}
              pathOptions={{ ...appearance, weight: 1.2 }}
              eventHandlers={{ click: () => onChoose(cell.center_latitude, cell.center_longitude) }}
            ><Tooltip>{cell.county_name ? `${cell.county_name} County · ` : ""}{status}</Tooltip></Rectangle>
          );
        })}
        <CircleMarker center={selected} radius={8} pathOptions={{ color: "#fff", weight: 3, fillColor: "#e8b44b", fillOpacity: 1 }}>
          <Tooltip permanent direction="top" offset={[0, -8]}>Selected location</Tooltip>
        </CircleMarker>
      </MapContainer>
      <div className="map-legend" aria-label="Map evidence legend">
        <span><i className="map-key strong" />Validated local</span>
        <span><i className="map-key limited" />Replay-supported local</span>
        <span><i className="map-key unavailable" />No published local result</span>
      </div>
      <p>Click inside the state. County lines are official Census boundaries; colored squares show the available evidence level.</p>
    </div>
  );
}
