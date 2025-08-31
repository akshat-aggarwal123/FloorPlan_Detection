import json
import uuid
from typing import Dict, List, Any

class FloorplanJSONConverter:
    def __init__(self, config):
        self.config = config
        
    def convert_to_json(self, walls, corners, doors, windows, rooms):
        """Convert extracted elements to JSON format matching your example"""
        
        # Generate wall connections between corners
        wall_connections = self._generate_wall_connections(walls, corners)
        
        # Create the floorplanner JSON structure
        floorplanner_data = {
            "version": "2.0.1a",
            "corners": corners,
            "walls": wall_connections,
            "rooms": self._format_rooms(rooms, corners),
            "wallTextures": [],
            "floorTextures": {},
            "newFloorTextures": self._generate_floor_textures(rooms),
            "carbonSheet": {},
            "boundary": {
                "points": [
                    {"x": -20, "y": -20, "elevation": 5},
                    {"x": 20, "y": -20, "elevation": 5},
                    {"x": 20, "y": 20, "elevation": 5},
                    {"x": -20, "y": 20, "elevation": 5}
                ],
                "style": {"color": "", "type": "texture", "colormap": "", "repeat": 3000}
            },
            "units": "m"
        }
        
        # Create items array with doors and windows
        items = self._create_items(doors, windows)
        
        # Create lighting
        lights = {
            "lights": [],
            "sunlight": [{
                "name": "SunLight",
                "position": {"x": 0, "y": 1000, "z": 1000},
                "intensity": 5,
                "color": 16777215,
                "shadow": True,
                "target": {"x": 0, "y": 0, "z": 0}
            }],
            "hemlight": [{
                "name": "HemisphereLight",
                "position": {"x": 0, "y": 0, "z": 0},
                "intensity": 0.6,
                "color": 16777215
            }],
            "amblight": [{
                "name": "AmbientLight",
                "position": {"x": 0, "y": 0, "z": 0},
                "intensity": 2.5,
                "color": 16775392
            }]
        }
        
        return {
            "floorplanner": floorplanner_data,
            "items": items,
            **lights
        }
    
    def _generate_wall_connections(self, walls, corners):
        """Generate wall connections between corners"""
        wall_connections = []
        corner_list = list(corners.keys())
        
        for i, wall in enumerate(walls):
            if i < len(corner_list) - 1:
                wall_data = {
                    "corner1": corner_list[i],
                    "corner2": corner_list[i + 1],
                    "frontTexture": self._default_wall_texture(),
                    "backTexture": self._default_wall_texture(),
                    "wallType": "STRAIGHT",
                    "a": {"x": wall['start']['x'], "y": wall['start']['y']},
                    "b": {"x": wall['end']['x'], "y": wall['end']['y']},
                    "thickness": 0.1
                }
                wall_connections.append(wall_data)
        
        return wall_connections
    
    def _default_wall_texture(self):
        """Default wall texture configuration"""
        return {
            "color": "#e6dcc1",
            "repeat": 300,
            "colormap": "textures/Wall/Indianwall/Wall.png",
            "normalmap": "textures/Wall/Indianwall/Wall_normal.jpg",
            "rotation": 0,
            "emissive": "#000000",
            "reflective": 0.5,
            "shininess": 0.5
        }
    
    def _format_rooms(self, rooms, corners):
        """Format rooms for JSON output"""
        formatted_rooms = {}
        corner_list = list(corners.keys())
        
        # Simple room formatting - in practice, you'd need more sophisticated room-corner mapping
        if len(corner_list) >= 4:
            room_key = ','.join(corner_list[:4])
            formatted_rooms[room_key] = {"name": "Living Room"}
            
            if len(corner_list) > 4:
                room_key2 = ','.join(corner_list[2:])
                formatted_rooms[room_key2] = {"name": "Bedroom"}
        
        return formatted_rooms
    
    def _generate_floor_textures(self, rooms):
        """Generate floor textures for rooms"""
        return {
            "default_room": {
                "name": "Herringbone_MarbleTiles",
                "repeat": 250,
                "colormap": "textures/Floor/HerringboneMarbleTiles/HerringboneMarbleTiles01_BaseColor.png",
                "normalmap": "textures/Floor/HerringboneMarbleTiles/HerringboneMarbleTiles01_Normal.png",
                "ambientmap": "textures/Floor/HerringboneMarbleTiles/HerringboneMarbleTiles01_AO.png",
                "roughnessmap": "textures/Floor/HerringboneMarbleTiles/HerringboneMarbleTiles01_Roughness.png",
                "color": "#FFFFFF",
                "emissive": "#000000",
                "reflective": 0.5,
                "shininess": 0.5
            }
        }
    
    def _create_items(self, doors, windows):
        """Create items array from doors and windows"""
        items = []
        
        for door in doors:
            items.append({
                "itemName": "Door 2",
                "thumbnail": "models/thumbnails/door1.png",
                "itemType": 7,
                "position": door['position'],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
                "size": door['size'],
                "modelURL": "models/glb/InWallFloorItem/door1.glb",
                "fixed": False,
                "resizable": False,
                "snap3D": True
            })
        
        for window in windows:
            items.append({
                "itemName": "Window 1",
                "thumbnail": "models/thumbnails/windows1.png",
                "itemType": 3,
                "position": window['position'],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
                "size": window['size'],
                "modelURL": "models/glb/InWallItem/windows1.glb",
                "fixed": False,
                "resizable": False,
                "snap3D": True
            })
        
        return items