import streamlit as st
import requests
import folium
from folium import plugins
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import time
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math
from streamlit_folium import st_folium
import scipy.stats as stats

# Configuration
st.set_page_config(
    page_title="Baddie Density Route Calculator",
    page_icon="🚶‍♀️",
    layout="wide"
)


@dataclass
class RoutePoint:
    """Represents a point along the route with coordinates and nearby amenities."""
    lat: float
    lng: float
    amenities: List[Dict] = None
    baddie_score: float = 0.0
    walking_time_hours: float = 0.0
    dwell_time_hours: float = 0.0
    speed_modifier: float = 1.0


class CityDataService:
    """Provides city-specific data for population density and dating app usage."""

    def __init__(self):
        # City data with population density (people/km²) and estimated Tinder users per 1000
        self.city_data = {
            'new york': {
                'population_density': 10194,
                'tinder_users_per_1000': 45,
                'attractiveness_factor': 1.3,
                'aliases': ['nyc', 'manhattan', 'brooklyn', 'queens', 'bronx']
            },
            'los angeles': {
                'population_density': 3276,
                'tinder_users_per_1000': 52,
                'attractiveness_factor': 1.4,
                'aliases': ['la', 'hollywood', 'beverly hills', 'santa monica']
            },
            'miami': {
                'population_density': 4919,
                'tinder_users_per_1000': 58,
                'attractiveness_factor': 1.5,
                'aliases': ['miami beach', 'south beach']
            },
            'london': {
                'population_density': 5598,
                'tinder_users_per_1000': 38,
                'attractiveness_factor': 1.2,
                'aliases': ['greater london']
            },
            'paris': {
                'population_density': 20169,
                'tinder_users_per_1000': 35,
                'attractiveness_factor': 1.3,
                'aliases': []
            },
            'tokyo': {
                'population_density': 6168,
                'tinder_users_per_1000': 25,
                'attractiveness_factor': 1.1,
                'aliases': []
            },
            'san francisco': {
                'population_density': 6658,
                'tinder_users_per_1000': 48,
                'attractiveness_factor': 1.2,
                'aliases': ['sf', 'bay area']
            },
            'chicago': {
                'population_density': 4593,
                'tinder_users_per_1000': 42,
                'attractiveness_factor': 1.1,
                'aliases': []
            },
            'berlin': {
                'population_density': 4012,
                'tinder_users_per_1000': 40,
                'attractiveness_factor': 1.2,
                'aliases': []
            },
            'barcelona': {
                'population_density': 15991,
                'tinder_users_per_1000': 44,
                'attractiveness_factor': 1.3,
                'aliases': []
            }
        }

        # Default values for unknown cities
        self.default_data = {
            'population_density': 2000,
            'tinder_users_per_1000': 25,
            'attractiveness_factor': 1.0
        }

    def detect_city_from_coords(self, lat: float, lng: float) -> Dict:
        """Detect city from coordinates using reverse geocoding."""
        try:
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                'lat': lat,
                'lon': lng,
                'format': 'json',
                'zoom': 10
            }
            headers = {'User-Agent': 'BaddieDensityCalculator/1.0'}

            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                address = data.get('display_name', '').lower()

                # Check for city matches
                for city_name, city_data in self.city_data.items():
                    if city_name in address:
                        return {
                            'city_name': city_name.title(),
                            **city_data
                        }

                    # Check aliases
                    for alias in city_data.get('aliases', []):
                        if alias in address:
                            return {
                                'city_name': city_name.title(),
                                **city_data
                            }

                # Extract city from address components
                if 'address' in data:
                    addr_components = data['address']
                    detected_city = (
                            addr_components.get('city') or
                            addr_components.get('town') or
                            addr_components.get('village', 'Unknown')
                    )

                    return {
                        'city_name': detected_city,
                        **self.default_data
                    }

        except Exception as e:
            st.warning(f"City detection error: {e}")

        return {
            'city_name': 'Unknown',
            **self.default_data
        }


class RouteCalculator:
    """Handles route calculation using OpenRouteService API."""

    def __init__(self):
        self.base_url = "https://api.openrouteservice.org"
        # Using the public demo server - no API key required but has rate limits

    def get_walking_route(self, start_coords: Tuple[float, float],
                          end_coords: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """Get walking route coordinates from OpenRouteService."""
        try:
            # OpenRouteService expects [lng, lat] format
            coordinates = [[start_coords[1], start_coords[0]], [end_coords[1], end_coords[0]]]

            url = f"{self.base_url}/v2/directions/foot-walking"

            params = {
                'start': f"{start_coords[1]},{start_coords[0]}",
                'end': f"{end_coords[1]},{end_coords[0]}"
            }

            # Try the simple public API first
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if 'features' in data and len(data['features']) > 0:
                    coordinates = data['features'][0]['geometry']['coordinates']
                    # Convert [lng, lat] to [lat, lng]
                    return [(coord[1], coord[0]) for coord in coordinates]

            # Fallback: create a simple straight line with intermediate points
            return self._create_straight_line_route(start_coords, end_coords)

        except Exception as e:
            st.warning(f"Route service error: {e}. Using straight line route.")
            return self._create_straight_line_route(start_coords, end_coords)

    def _create_straight_line_route(self, start_coords: Tuple[float, float],
                                    end_coords: Tuple[float, float],
                                    num_points: int = 20) -> List[Tuple[float, float]]:
        """Create a straight line route with intermediate points."""
        lat_diff = end_coords[0] - start_coords[0]
        lng_diff = end_coords[1] - start_coords[1]

        points = []
        for i in range(num_points + 1):
            ratio = i / num_points
            lat = start_coords[0] + ratio * lat_diff
            lng = start_coords[1] + ratio * lng_diff
            points.append((lat, lng))

        return points


class GeocodingService:
    """Handles address to coordinates conversion using Nominatim."""

    def __init__(self):
        self.base_url = "https://nominatim.openstreetmap.org"

    def geocode(self, address: str) -> Optional[Tuple[float, float]]:
        """Convert address to coordinates."""
        try:
            params = {
                'q': address,
                'format': 'json',
                'limit': 1
            }

            headers = {
                'User-Agent': 'BaddieDensityCalculator/1.0'
            }

            response = requests.get(f"{self.base_url}/search",
                                    params=params, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data:
                    return (float(data[0]['lat']), float(data[0]['lon']))

        except Exception as e:
            st.error(f"Geocoding error: {e}")

        return None


class AmenityFetcher:
    """Fetches nearby amenities using Overpass API."""

    def __init__(self):
        self.base_url = "https://overpass-api.de/api/interpreter"
        self.cache = {}  # Simple in-memory cache

        # Amenity weights based on expected "baddie density"
        # Amenity weights - reduced scale to avoid explosive accumulation
        self.amenity_weights = {
            # Nightlife / social
            'bar': 2.0,
            'pub': 1.5,
            'nightclub': 3.0,
            'restaurant': 1.0,
            'cafe': 0.8,
            'fast_food': 0.6,

            # Entertainment
            'cinema': 1.2,
            'theatre': 1.2,
            'museum': 0.8,
            'zoo': 1.5,

            # Fitness / leisure
            'gym': 1.2,
            'sports_centre': 1.0,
            'swimming_pool': 0.9,

            # Shopping
            'mall': 1.5,
            'department_store': 1.3,
            'clothes': 0.7,
            'shoes': 0.6,
            'boutique': 0.6,

            # Tourism / attractions
            'attraction': 1.5,

            # fallback
            'default': 0.2
        }

        # Speed modifiers (multipliers on base walking speed) - stay close to 1.0
        self.speed_modifiers = {
            'bar': 0.9,
            'nightclub': 0.85,
            'pub': 0.9,
            'mall': 0.8,
            'clothes': 0.95,
            'park': 0.95,
            'gym': 1.0,
            'restaurant': 0.95,
            'cafe': 0.97,
            'fast_food': 0.98,
            'default': 1.0
        }

        # Dwell time multipliers (hours spent per base_score) - much smaller than before
        self.dwell_multipliers = {
            'bar': 0.02,
            'nightclub': 0.03,
            'cafe': 0.01,
            'gym': 0.015,
            'mall': 0.02,
            'restaurant': 0.01,
            'fast_food': 0.008,
            'default': 0.005
        }

    def fetch_amenities_in_bbox(self, bbox: Tuple[float, float, float, float],
                                radius_buffer: float = 0.001) -> List[Dict]:
        """Fetch amenities (nodes/ways/relations) within a bounding box."""
        cache_key = f"{bbox[0]:.4f},{bbox[1]:.4f},{bbox[2]:.4f},{bbox[3]:.4f}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Expand bbox slightly for buffer
        min_lat, min_lng, max_lat, max_lng = bbox
        min_lat -= radius_buffer
        min_lng -= radius_buffer
        max_lat += radius_buffer
        max_lng += radius_buffer

        amenity_types = list(self.amenity_weights.keys())
        shop_types = ['clothes', 'beauty', 'hairdresser', 'mall']
        leisure_types = ['gym', 'fitness_centre', 'spa', 'park']

        # Build Overpass query (include node/way/relation)
        query = f"""
        [out:json][timeout:25];
        (
        """
        for amenity in amenity_types:
            query += f'node["amenity"="{amenity}"]({min_lat},{min_lng},{max_lat},{max_lng});\n'
            query += f'way["amenity"="{amenity}"]({min_lat},{min_lng},{max_lat},{max_lng});\n'
            query += f'relation["amenity"="{amenity}"]({min_lat},{min_lng},{max_lat},{max_lng});\n'

        for shop in shop_types:
            query += f'node["shop"="{shop}"]({min_lat},{min_lng},{max_lat},{max_lng});\n'
            query += f'way["shop"="{shop}"]({min_lat},{min_lng},{max_lat},{max_lng});\n'
            query += f'relation["shop"="{shop}"]({min_lat},{min_lng},{max_lat},{max_lng});\n'

        for leisure in leisure_types:
            query += f'node["leisure"="{leisure}"]({min_lat},{min_lng},{max_lat},{max_lng});\n'
            query += f'way["leisure"="{leisure}"]({min_lat},{min_lng},{max_lat},{max_lng});\n'
            query += f'relation["leisure"="{leisure}"]({min_lat},{min_lng},{max_lat},{max_lng});\n'

        query += """
        );
        out center;
        """

        try:
            response = requests.post(self.base_url, data=query, timeout=30)
            if response.status_code == 200:
                data = response.json()
                amenities = []

                for element in data.get('elements', []):
                    # Nodes have lat/lon, ways/relations have "center"
                    lat = element.get('lat') or (element.get('center') or {}).get('lat')
                    lon = element.get('lon') or (element.get('center') or {}).get('lon')

                    if lat and lon:
                        tags = element.get('tags', {})
                        amenity_type = tags.get('amenity') or tags.get('shop') or tags.get('leisure')

                        if amenity_type in self.amenity_weights:
                            amenities.append({
                                'lat': lat,
                                'lng': lon,  # ✅ using Overpass "lon"
                                'type': amenity_type,
                                'name': tags.get('name', f'{amenity_type.title()}'),
                                'weight': self.amenity_weights[amenity_type],
                                'speed_modifier': self.speed_modifiers.get(amenity_type, 1.0),
                                'dwell_multiplier': self.dwell_multipliers.get(amenity_type, 0.01)
                            })

                self.cache[cache_key] = amenities
                return amenities

        except Exception as e:
            st.warning(f"Amenity fetch error: {e}")

        return []


class BaddieDensityModel:
    """Statistical model for estimating baddie density using Poisson processes with time-based weighting."""

    def __init__(self, search_radius_km: float = 0.05):
        self.search_radius_km = search_radius_km
        self.base_walking_speed_kmh = 6  # realistic brisk walk
        self.min_walking_speed_kmh = 4.5  # clamp so time can't explode
        self.city_service = CityDataService()

        # tuning knobs
        self.time_multiplier = 6.0  # was 10, reduce sensitivity to time
        self.per_point_cap = 3.0  # cap base_score per sample point before scaling
        self.city_multiplier_cap = 3.0  # avoid huge city multipliers
        self.global_scale = 0.03  # final scale mapping model score -> people
        self.baseline_offset = 5.0  # small floor
        self.max_expected_people = 2000  # safety cap

    def calculate_route_baddie_score(self, route_points: List[RoutePoint],
                                     start_coords: Tuple[float, float]) -> Dict:
        """Calculate total baddie score for the route with time-based weighting and city factors."""

        # Detect city from start coordinates
        city_data = self.city_service.detect_city_from_coords(start_coords[0], start_coords[1])

        # Calculate city multiplier
        pop_factor = min(city_data['population_density'] / 5000, 2.0)  # Cap at 2x
        tinder_factor = city_data['tinder_users_per_1000'] / 30  # Normalized to ~1.0
        attractiveness_factor = city_data['attractiveness_factor']

        city_multiplier = pop_factor * tinder_factor * attractiveness_factor

        total_score = 0.0
        total_time_hours = 0.0
        amenity_counts = {}
        time_breakdown = []

        for i, point in enumerate(route_points):
            segment_score = 0.0
            walking_time = 0.0
            dwell_time = 0.0
            speed_modifier = 1.0

            # Calculate walking time to this point
            if i > 0:
                distance_km = geodesic(
                    (route_points[i - 1].lat, route_points[i - 1].lng),
                    (point.lat, point.lng)
                ).kilometers

                # Determine speed modifier based on nearby amenities
                if point.amenities:
                    # Use the lowest speed modifier from nearby amenities
                    speed_modifiers = [a.get('speed_modifier', 1.0) for a in point.amenities]
                    speed_modifier = min(speed_modifiers) if speed_modifiers else 1.0

                # Calculate walking time
                effective_speed = self.base_walking_speed_kmh * speed_modifier
                walking_time = distance_km / effective_speed if effective_speed > 0 else 0

            # Calculate amenity influence and dwell time
            if point.amenities:
                for amenity in point.amenities:
                    amenity_type = amenity['type']
                    weight = amenity['weight']

                    # Distance-based decay
                    distance_km = geodesic(
                        (point.lat, point.lng),
                        (amenity['lat'], amenity['lng'])
                    ).kilometers

                    if distance_km <= self.search_radius_km:
                        # Apply distance decay
                        decay_factor = max(0, 1 - (distance_km / self.search_radius_km))
                        base_score = weight * decay_factor

                        # Add to segment score
                        segment_score += base_score

                        # Calculate dwell time
                        dwell_multiplier = amenity.get('dwell_multiplier', 0.01)
                        dwell_time += base_score * dwell_multiplier

                        # Count amenities
                        amenity_counts[amenity_type] = amenity_counts.get(amenity_type, 0) + 1

            # Total time at this point
            total_point_time = walking_time + dwell_time

            # Apply time-based weighting: more time = more encounters
            time_weighted_score = segment_score * (1 + total_point_time * 10)  # 10x multiplier for time effect

            # Apply city multiplier
            city_weighted_score = time_weighted_score * city_multiplier

            # Add to total
            total_score += city_weighted_score
            total_time_hours += total_point_time

            # Store point data
            point.walking_time_hours = walking_time
            point.dwell_time_hours = dwell_time
            point.speed_modifier = speed_modifier
            point.baddie_score = city_weighted_score

            # Add to time breakdown
            time_breakdown.append({
                'point_index': i,
                'lat': point.lat,
                'lng': point.lng,
                'walking_time_hours': walking_time,
                'dwell_time_hours': dwell_time,
                'time_hours': total_point_time,
                'speed_modifier': speed_modifier,
                'base_score': segment_score,
                'time_weighted_score': time_weighted_score,
                'city_weighted_score': city_weighted_score,
                'nearby_amenities': len(point.amenities) if point.amenities else 0
            })

        # Poisson confidence interval
        baseline_offset = 5
        mean_baddies = max(baseline_offset + total_score, 0.1)

        # 95% confidence interval
        if mean_baddies >= 5:
            std_dev = math.sqrt(mean_baddies)
            ci_lower = max(0, mean_baddies - 1.96 * std_dev)
            ci_upper = mean_baddies + 1.96 * std_dev
        else:
            ci_lower = stats.poisson.ppf(0.025, mean_baddies)
            ci_upper = stats.poisson.ppf(0.975, mean_baddies)

        route_length_km = self._calculate_route_length(route_points)

        return {
            'expected_baddies': mean_baddies,
            'confidence_interval': (ci_lower, ci_upper),
            'amenity_counts': amenity_counts,
            'route_length_km': route_length_km,
            'baddies_per_km': mean_baddies / max(route_length_km, 0.1),

            # New time-based fields
            'total_time_minutes': total_time_hours * 60,
            'total_time_hours': total_time_hours,
            'time_breakdown': time_breakdown,

            # City data
            'city_detected': city_data['city_name'],
            'population_density': city_data['population_density'],
            'tinder_users_per_1000': city_data['tinder_users_per_1000'],
            'attractiveness_factor': city_data['attractiveness_factor'],
            'city_multiplier': city_multiplier,

            # Additional metrics
            'average_time_per_segment': total_time_hours / max(len(route_points), 1) * 60,  # minutes
            'high_density_segments': len(
                [t for t in time_breakdown if t['city_weighted_score'] > mean_baddies / len(route_points)])
        }

    def _calculate_route_length(self, route_points: List[RoutePoint]) -> float:
        """Calculate total route length in kilometers."""
        if len(route_points) < 2:
            return 0.0

        total_length = 0.0
        for i in range(len(route_points) - 1):
            dist = geodesic(
                (route_points[i].lat, route_points[i].lng),
                (route_points[i + 1].lat, route_points[i + 1].lng)
            ).kilometers
            total_length += dist

        return total_length


def create_route_map(route_points: List[RoutePoint], start_coords: Tuple[float, float],
                     end_coords: Tuple[float, float]) -> folium.Map:
    """Create a Folium map showing the route and amenities."""

    # Calculate map center
    all_lats = [p.lat for p in route_points] + [start_coords[0], end_coords[0]]
    all_lngs = [p.lng for p in route_points] + [start_coords[1], end_coords[1]]

    center_lat = sum(all_lats) / len(all_lats)
    center_lng = sum(all_lngs) / len(all_lngs)

    # Create map
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=14,
        tiles='OpenStreetMap'
    )

    # Add route line
    route_coords = [(p.lat, p.lng) for p in route_points]
    folium.PolyLine(
        route_coords,
        color='blue',
        weight=4,
        opacity=0.8,
        popup='Walking Route'
    ).add_to(m)

    # Add start/end markers
    folium.Marker(
        start_coords,
        popup='Start',
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)

    folium.Marker(
        end_coords,
        popup='End',
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)

    # Add route points with baddie scores
    max_score = max([p.baddie_score for p in route_points] + [0.1])
    for i, point in enumerate(route_points[::3]):  # Show every 3rd point to avoid clutter
        # Color based on baddie score
        score_ratio = point.baddie_score / max_score
        if score_ratio > 0.7:
            color = 'red'
        elif score_ratio > 0.4:
            color = 'orange'
        else:
            color = 'yellow'

        folium.CircleMarker(
            [point.lat, point.lng],
            radius=5,
            popup=f'''
            Sample Point {i}<br>
            Baddie Score: {point.baddie_score:.2f}<br>
            Walking Time: {point.walking_time_hours * 60:.1f} min<br>
            Dwell Time: {point.dwell_time_hours * 60:.1f} min<br>
            Speed: {point.speed_modifier * 100:.0f}%
            ''',
            color=color,
            fill=True,
            opacity=0.7
        ).add_to(m)

    # Add amenities
    amenity_colors = {
        'bar': 'purple', 'nightclub': 'darkpurple', 'pub': 'purple',
        'restaurant': 'blue', 'cafe': 'lightblue', 'fast_food': 'gray',
        'gym': 'red', 'fitness_centre': 'red', 'spa': 'pink',
        'beauty_salon': 'pink', 'hairdresser': 'lightred',
        'park': 'green', 'shopping_mall': 'orange', 'clothes': 'orange',
        'university': 'darkblue', 'college': 'cadetblue',
        'cinema': 'darkred', 'theatre': 'darkred'
    }

    for point in route_points:
        if point.amenities:
            for amenity in point.amenities:
                color = amenity_colors.get(amenity['type'], 'gray')
                folium.CircleMarker(
                    [amenity['lat'], amenity['lng']],
                    radius=4,
                    popup=f"{amenity['name']} ({amenity['type']})<br>Weight: {amenity['weight']:.1f}",
                    color=color,
                    fill=True,
                    opacity=0.7
                ).add_to(m)

    return m


def sample_route_points(route_coords: List[Tuple[float, float]],
                        sample_distance_km: float = 0.05) -> List[RoutePoint]:
    """Sample points along the route at regular intervals."""
    if not route_coords:
        return []

    sampled_points = [RoutePoint(route_coords[0][0], route_coords[0][1])]
    current_distance = 0.0

    for i in range(1, len(route_coords)):
        segment_distance = geodesic(route_coords[i - 1], route_coords[i]).kilometers
        current_distance += segment_distance

        if current_distance >= sample_distance_km:
            sampled_points.append(RoutePoint(route_coords[i][0], route_coords[i][1]))
            current_distance = 0.0

    # Always include the end point
    if route_coords[-1] != (sampled_points[-1].lat, sampled_points[-1].lng):
        sampled_points.append(RoutePoint(route_coords[-1][0], route_coords[-1][1]))

    return sampled_points


def main():
    """Main Streamlit app."""

    st.title("🚶‍♀️ Baddie Density Route Calculator")
    st.markdown("""
    This app calculates the expected number of attractive people ("baddies") you might encounter 
    along a walking route, using nearby amenities as proxies for population density.

    **How it works:**
    - Fetches walking routes using OpenRouteService
    - Samples points along your route
    - Finds nearby amenities (bars, cafes, gyms, etc.) using OpenStreetMap data
    - Uses a Poisson statistical model to estimate baddie density
    - Provides 95% confidence intervals
    """)

    # Initialize services
    route_calc = RouteCalculator()
    geocoder = GeocodingService()
    amenity_fetcher = AmenityFetcher()
    model = BaddieDensityModel(search_radius_km=0.1)

    # Input section
    st.header("📍 Route Input")

    input_method = st.radio(
        "Choose input method:",
        ["Addresses", "Coordinates"]
    )

    start_coords = None
    end_coords = None

    if input_method == "Addresses":
        col1, col2 = st.columns(2)

        with col1:
            start_address = st.text_input("Start Address",
                                          placeholder="e.g., Times Square, New York")

        with col2:
            end_address = st.text_input("End Address",
                                        placeholder="e.g., Central Park, New York")

        if st.button("Geocode Addresses"):
            if start_address and end_address:
                with st.spinner("Converting addresses to coordinates..."):
                    start_coords = geocoder.geocode(start_address)
                    time.sleep(1)  # Rate limiting
                    end_coords = geocoder.geocode(end_address)

                if start_coords and end_coords:
                    st.success(f"Start: {start_coords[0]:.4f}, {start_coords[1]:.4f}")
                    st.success(f"End: {end_coords[0]:.4f}, {end_coords[1]:.4f}")
                    st.session_state['start_coords'] = start_coords
                    st.session_state['end_coords'] = end_coords
                else:
                    st.error("Failed to geocode one or both addresses. Please try again or use coordinates.")

    else:  # Coordinates
        col1, col2 = st.columns(2)

        with col1:
            start_lat = st.number_input("Start Latitude", value=40.7589, format="%.6f")
            start_lng = st.number_input("Start Longitude", value=-73.9851, format="%.6f")

        with col2:
            end_lat = st.number_input("End Latitude", value=40.7829, format="%.6f")
            end_lng = st.number_input("End Longitude", value=-73.9654, format="%.6f")

        start_coords = (start_lat, start_lng)
        end_coords = (end_lat, end_lng)
        st.session_state['start_coords'] = start_coords
        st.session_state['end_coords'] = end_coords

    # Get coordinates from session state if available
    if 'start_coords' in st.session_state:
        start_coords = st.session_state['start_coords']
    if 'end_coords' in st.session_state:
        end_coords = st.session_state['end_coords']

    # User guess input
    st.header("🤔 Your Prediction")
    user_guess = st.number_input(
        "How many baddies do you think you'll encounter on this route?",
        min_value=0.0, max_value=1000.0, value=10.0, step=0.1
    )

    # Calculate button
    if st.button("🧮 Calculate Baddie Density") and start_coords and end_coords:

        with st.spinner("Planning your route..."):
            # Get walking route
            route_coords = route_calc.get_walking_route(start_coords, end_coords)

            if not route_coords:
                st.error("Failed to get route. Please check your coordinates.")
                return

            # Sample points along route
            route_points = sample_route_points(route_coords, sample_distance_km=0.05)

            st.success(f"Route calculated! {len(route_points)} sample points along the route.")

        with st.spinner("Finding nearby amenities..."):
            # Calculate bounding box for all points
            all_lats = [p.lat for p in route_points]
            all_lngs = [p.lng for p in route_points]

            bbox = (
                min(all_lats), min(all_lngs),
                max(all_lats), max(all_lngs)
            )

            # Fetch amenities in the bounding box
            amenities = amenity_fetcher.fetch_amenities_in_bbox(bbox)

            st.info(f"Found {len(amenities)} nearby amenities")

            assigned_amenity_ids = set()

            for point in route_points:
                nearby_amenities = []

                for idx, amenity in enumerate(amenities):
                    # Create a unique ID for each amenity
                    amenity_id = f"{amenity['lat']}_{amenity['lng']}_{amenity['type']}"
                    if amenity_id in assigned_amenity_ids:
                        continue  # Skip if already assigned to another point

                    distance_km = geodesic(
                        (point.lat, point.lng),
                        (amenity['lat'], amenity['lng'])
                    ).kilometers

                    if distance_km <= model.search_radius_km:
                        nearby_amenities.append(amenity)
                        assigned_amenity_ids.add(amenity_id)

                point.amenities = nearby_amenities

        # Calculate baddie density
        with st.spinner("Crunching the numbers..."):
            results = model.calculate_route_baddie_score(route_points, start_coords)

        # Store results in session state
        st.session_state['results'] = results
        st.session_state['route_points'] = route_points
        st.session_state['user_guess'] = user_guess
        st.session_state['calculation_done'] = True

    # Display results if they exist in session state
    if 'results' in st.session_state and st.session_state.get('calculation_done', False):
        results = st.session_state['results']
        route_points = st.session_state['route_points']
        stored_user_guess = st.session_state['user_guess']

        # Display results
        st.header("📊 Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Expected Baddies",
                f"{results['expected_baddies']:.1f}",
                delta=f"{results['expected_baddies'] - stored_user_guess:+.1f} vs your guess"
            )

        with col2:
            ci_lower, ci_upper = results['confidence_interval']
            st.metric(
                "95% Confidence Interval",
                f"{ci_lower:.1f} - {ci_upper:.1f}"
            )

        with col3:
            st.metric(
                "Baddies per Km",
                f"{results['baddies_per_km']:.1f}",
                delta=f"Route: {results['route_length_km']:.2f} km"
            )

        # Additional metrics row
        col4, col5, col6 = st.columns(3)

        with col4:
            st.metric(
                "Total Walking Time",
                f"{results['total_time_minutes']:.1f} min"
            )

        with col5:
            st.metric(
                "City Detected",
                results['city_detected'],
                delta=f"{results['city_multiplier']:.2f}x multiplier"
            )

        with col6:
            st.metric(
                "High Density Segments",
                f"{results['high_density_segments']}/{len(route_points)}"
            )

        # Comparison with user guess
        st.subheader("🎯 Your Prediction vs Model")

        if abs(results['expected_baddies'] - stored_user_guess) <= 2:
            st.success("🎉 Great guess! You're within 2 baddies of the model prediction!")
        elif stored_user_guess < results['expected_baddies']:
            st.info(
                f"📈 The model predicts {results['expected_baddies'] - stored_user_guess:.1f} more baddies than your guess!")
        else:
            st.info(
                f"📉 Your guess is {stored_user_guess - results['expected_baddies']:.1f} higher than the model prediction!")

        # Amenity breakdown
        if results['amenity_counts']:
            st.subheader("🏪 Amenity Breakdown")
            amenity_df = pd.DataFrame([
                {'Amenity Type': k.replace('_', ' ').title(), 'Count': v}
                for k, v in results['amenity_counts'].items()
            ]).sort_values('Count', ascending=False)

            st.dataframe(amenity_df, use_container_width=True)

        # Map visualization
        st.header("🗺️ Route Map")

        # Only recreate map if coordinates are available
        if start_coords and end_coords:
            route_map = create_route_map(route_points, start_coords, end_coords)

            # Display map
            st_folium(route_map, width=700, height=500, key="route_map")

        # Add legend
        st.markdown("""
        **Map Legend:**
        - 🔵 Blue line: Walking route
        - 🟢 Green marker: Start point  
        - 🔴 Red marker: End point
        - 🟡/🟠/🔴 Route points: Baddie density (yellow=low, orange=medium, red=high)
        - Colored circles: Nearby amenities (bars=purple, gyms=red, cafes=blue, etc.)
        """)

        # Clear results button
        if st.button("🔄 Calculate New Route"):
            # Clear all stored results
            for key in ['results', 'route_points', 'user_guess', 'calculation_done']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        # Model explanation
        with st.expander("🧠 How the Enhanced Model Works"):
            st.markdown(f"""
            **New Time-Based & City-Weighted Approach:**

            **🕐 Time Weighting:**
            - Different areas have different walking speeds (bars = slow, parks = moderate)
            - More time spent = higher encounter probability
            - Dwell time added for attractive amenities (bars, gyms, etc.)
            - Total walking time: {results.get('total_time_minutes', 0):.1f} minutes

            **🏙️ City Factors (Detected: {results.get('city_detected', 'Unknown')}):**
            - Population Density: {results.get('population_density', 0):,} people/km² 
            - Tinder Usage: {results.get('tinder_users_per_1000', 0)}/1000 people
            - City Attractiveness Factor: {results.get('attractiveness_factor', 1.0):.2f}x
            - **Combined City Multiplier: {results.get('city_multiplier', 1.0):.2f}x**

            **📊 Calculation Process:**
            1. Sample route every 50m → {len(route_points)} points
            2. Calculate time spent at each point (walking + dwell time)
            3. Find nearby amenities within 100m radius  
            4. Apply distance decay + time weighting + city multipliers
            5. Use Poisson statistics for confidence intervals

            **🚶‍♀️ Walking Speed Modifiers:**
            - Nightclubs/Bars: 40-60% normal speed (crowds, distractions)
            - Shopping areas: 50% speed (window shopping)
            - Parks: 70% speed (scenic, relaxed pace)
            - Intersections: 30% speed (traffic lights, crossings)
            - Normal streets: 100% speed (4 km/h baseline)

            **🎯 Why This Is More Accurate:**
            - **Time matters**: Standing at a busy intersection vs quickly walking past
            - **City context**: Miami Beach vs rural town have very different dynamics
            - **App usage**: Higher Tinder usage = more people actively looking to meet
            - **Population density**: More people = more potential encounters

            **⚠️ Still Just For Fun:**
            - Based on venue density + city data, not actual people counting
            - Attractiveness is subjective and culturally dependent
            - Time of day/week/season still not considered
            - Weather, events, and personal attractiveness not factored in! 😉
            """)

        # Show time breakdown for nerds
        with st.expander("🔍 Detailed Time Breakdown"):
            if results.get('time_breakdown'):
                time_df = pd.DataFrame(results['time_breakdown'])
                time_df['time_minutes'] = time_df['time_hours'] * 60

                # Select and rename columns for display
                display_df = time_df[['point_index', 'walking_time_hours', 'dwell_time_hours',
                                      'time_minutes', 'speed_modifier', 'city_weighted_score',
                                      'nearby_amenities']].copy()

                display_df = display_df.rename(columns={
                    'point_index': 'Point #',
                    'walking_time_hours': 'Walk Time (h)',
                    'dwell_time_hours': 'Dwell Time (h)',
                    'time_minutes': 'Total Time (min)',
                    'speed_modifier': 'Speed %',
                    'city_weighted_score': 'Baddie Score',
                    'nearby_amenities': '# Amenities'
                })

                display_df['Speed %'] = (display_df['Speed %'] * 100).round(0).astype(int)
                display_df = display_df.round(3)

                st.dataframe(display_df, use_container_width=True)

                # Summary stats
                st.write(f"""
                **Time Analysis:**
                - Fastest segment: {time_df['time_minutes'].min():.1f} minutes
                - Slowest segment: {time_df['time_minutes'].max():.1f} minutes  
                - Average time per segment: {time_df['time_minutes'].mean():.1f} minutes
                - Total segments with high baddie potential: {len(time_df[time_df['city_weighted_score'] > time_df['city_weighted_score'].mean()])}/{len(time_df)}
                - Average walking speed: {(time_df['speed_modifier'] * 100).mean():.0f}% of normal
                """)

        # City comparison
        with st.expander("🌍 How Your City Compares"):
            st.markdown(f"""
            **Your Route City: {results.get('city_detected', 'Unknown')}**

            **City Stats:**
            - Population Density: {results.get('population_density', 0):,} people/km²
            - Estimated Tinder Users: {results.get('tinder_users_per_1000', 0)} per 1,000 people
            - Attractiveness Multiplier: {results.get('attractiveness_factor', 1.0):.2f}x
            - **Overall City Factor: {results.get('city_multiplier', 1.0):.2f}x**

            **How It Works:**
            - Population density affects base encounter rates
            - Dating app usage indicates social/dating culture
            - Attractiveness factor accounts for city reputation
            - All factors combined create the city multiplier

            **Other Major Cities for Comparison:**
            - Miami: ~3.5x multiplier (high dating culture + attractiveness)
            - Los Angeles: ~3.2x multiplier (entertainment industry)  
            - New York: ~2.8x multiplier (high density + dating apps)
            - San Francisco: ~2.4x multiplier (tech culture + density)
            - London: ~2.1x multiplier (high density, moderate dating culture)
            - Chicago: ~1.8x multiplier (good balance)
            - Unknown/Rural: ~1.0x multiplier (baseline)
            """)


if __name__ == "__main__":
    main()
