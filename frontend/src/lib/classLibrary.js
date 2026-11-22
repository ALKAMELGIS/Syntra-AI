// Ready-made object classes, by theme. Each has the drawing tool that suits it
// (a key of LABEL_TOOLS) and the outline the AI tools fit to it (a Refine shape
// in app/engine/shapes.py: circle, rectangle, right_angles, smooth, polygon).
const g = (group, rows) => rows.map(([name, tool, shape]) => ({ group, name, tool, shape }));

export const CLASS_GROUPS = ["Agriculture", "Urban", "Environment", "Infrastructure", "Transportation", "Natural features"];

export const SHAPE_NAMES = { circle: "Circle", rectangle: "Rotated box", box: "Box", right_angles: "Right angles", smooth: "Smooth", polygon: "Polygon" };
export const TOOL_NAMES = { square: "Square", rect: "Rectangle", polygon: "Polygon", circle: "Circle", freehand: "Freehand", click: "Click object" };

export const CLASS_LIBRARY = [
  ...g("Agriculture", [
    ["Tree", "square", "circle"], ["Palm tree", "square", "circle"], ["Date palm", "square", "circle"], ["Fruit tree", "square", "circle"],
    ["Field", "polygon", "polygon"], ["Crop row", "rect", "rectangle"], ["Greenhouse", "rect", "rectangle"], ["Net house", "rect", "rectangle"],
    ["Irrigation pivot", "circle", "circle"], ["Tractor", "rect", "rectangle"], ["Farm machinery", "rect", "rectangle"],
    ["Livestock", "square", "polygon"], ["Farm building", "polygon", "right_angles"],
  ]),
  ...g("Urban", [
    ["Building", "polygon", "right_angles"], ["House", "polygon", "right_angles"], ["Industrial building", "polygon", "right_angles"],
    ["Commercial building", "polygon", "right_angles"], ["Road", "polygon", "smooth"], ["Vehicle", "rect", "rectangle"], ["Car", "rect", "rectangle"],
    ["Truck", "rect", "rectangle"], ["Bus", "rect", "rectangle"], ["Parking area", "polygon", "right_angles"],
    ["Construction site", "polygon", "polygon"], ["Solar panel", "rect", "rectangle"], ["Swimming pool", "polygon", "polygon"],
  ]),
  ...g("Environment", [
    ["Water tank", "circle", "circle"], ["Water body", "polygon", "smooth"], ["River", "polygon", "smooth"], ["Lake", "polygon", "smooth"],
    ["Pond", "polygon", "smooth"], ["Waterlogged area", "polygon", "smooth"], ["Waste dump", "polygon", "polygon"], ["Landfill", "polygon", "polygon"],
    ["Burned area", "polygon", "smooth"], ["Mining site", "polygon", "polygon"], ["Bare soil", "polygon", "smooth"], ["Vegetation cluster", "polygon", "smooth"],
  ]),
  ...g("Infrastructure", [
    ["Power pole", "square", "polygon"], ["Transmission tower", "square", "polygon"], ["Streetlight", "square", "polygon"], ["Bridge", "rect", "rectangle"],
    ["Road intersection", "polygon", "polygon"], ["Pipeline", "polygon", "smooth"], ["Communication tower", "square", "polygon"],
    ["Oil tank", "circle", "circle"], ["Railway", "polygon", "smooth"], ["Drainage structure", "polygon", "polygon"],
  ]),
  ...g("Transportation", [
    ["Aircraft", "rect", "rectangle"], ["Ship", "rect", "rectangle"], ["Boat", "rect", "rectangle"], ["Train", "rect", "rectangle"],
    ["Heavy equipment", "rect", "rectangle"],
  ]),
  ...g("Natural features", [
    ["Forest patch", "polygon", "smooth"], ["Rock formation", "polygon", "polygon"], ["Coastline", "polygon", "smooth"],
    ["Sand dune", "polygon", "smooth"], ["Natural vegetation", "polygon", "smooth"],
  ]),
];

export const classSlug = (name) => name.trim().toLowerCase().replace(/\s+/g, "_");
