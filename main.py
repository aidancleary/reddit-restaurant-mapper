import praw
import os
import re
import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from flask import Flask, render_template, request
import requests
from urllib.parse import quote
from anthropic import Anthropic
import json
import logging
from bs4 import BeautifulSoup
import re
from fake_useragent import UserAgent
import time

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# After your imports but before app = Flask(__name__)

CITY_SUBREDDITS = {
    # Food-specific subreddits
    'bostonfood': 'Boston, MA',
    'foodnyc': 'New York, NY',
    'chicagofood': 'Chicago, IL',
    'sandiegofood': 'San Diego, CA',
    'lafood': 'Los Angeles, CA',
    'dcfood': 'Washington, DC',
    'phillyfood': 'Philadelphia, PA',
    'houstonfood': 'Houston, TX',

    # General city subreddits that often discuss food
    'sfbayarea': 'San Francisco, CA',
    'nyc': 'New York, NY',
    'boston': 'Boston, MA',
    'chicago': 'Chicago, IL',
    'seattle': 'Seattle, WA',
    'portland': 'Portland, OR',
    'atlanta': 'Atlanta, GA',
    'miami': 'Miami, FL',
    'denver': 'Denver, CO',
    'austin': 'Austin, TX'
}
NYC_NEIGHBORHOODS = {
    # Manhattan
    'les': 'Lower East Side, Manhattan, NY',
    'ues': 'Upper East Side, Manhattan, NY',
    'uws': 'Upper West Side, Manhattan, NY',
    'east village': 'East Village, Manhattan, NY',
    'west village': 'West Village, Manhattan, NY',
    'chinatown': 'Chinatown, Manhattan, NY',
    'tribeca': 'Tribeca, Manhattan, NY',
    'soho': 'SoHo, Manhattan, NY',
    'noho': 'NoHo, Manhattan, NY',
    'fidi': 'Financial District, Manhattan, NY',
    'harlem': 'Harlem, Manhattan, NY',
    'washington heights': 'Washington Heights, Manhattan, NY',
    'hells kitchen': "Hell's Kitchen, Manhattan, NY",
    'murray hill': 'Murray Hill, Manhattan, NY',
    'gramercy': 'Gramercy, Manhattan, NY',
    'chelsea': 'Chelsea, Manhattan, NY',
    'midtown': 'Midtown, Manhattan, NY',

    # Brooklyn
    'williamsburg': 'Williamsburg, Brooklyn, NY',
    'greenpoint': 'Greenpoint, Brooklyn, NY',
    'dumbo': 'DUMBO, Brooklyn, NY',
    'bushwick': 'Bushwick, Brooklyn, NY',
    'park slope': 'Park Slope, Brooklyn, NY',
    'fort greene': 'Fort Greene, Brooklyn, NY',
    'clinton hill': 'Clinton Hill, Brooklyn, NY',
    'crown heights': 'Crown Heights, Brooklyn, NY',
    'prospect heights': 'Prospect Heights, Brooklyn, NY',
    'bed stuy': 'Bedford-Stuyvesant, Brooklyn, NY',
    'bedford stuyvesant': 'Bedford-Stuyvesant, Brooklyn, NY',
    'carroll gardens': 'Carroll Gardens, Brooklyn, NY',
    'cobble hill': 'Cobble Hill, Brooklyn, NY',
    'boerum hill': 'Boerum Hill, Brooklyn, NY',

    # Queens
    'astoria': 'Astoria, Queens, NY',
    'lic': 'Long Island City, Queens, NY',
    'long island city': 'Long Island City, Queens, NY',
    'jackson heights': 'Jackson Heights, Queens, NY',
    'flushing': 'Flushing, Queens, NY',
    'forest hills': 'Forest Hills, Queens, NY'
}

# Restaurant-related constants
RESTAURANT_INDICATORS = {
    # Types of establishments
    'restaurant', 'cafe', 'pizzeria', 'diner', 'bistro', 'grill', 
    'eatery', 'bar', 'pub', 'tavern', 'steakhouse', 'kitchen',
    'bakery', 'trattoria', 'osteria', 'chophouse',

    # Cuisine types
    'sushi', 'ramen', 'izakaya', 'bbq', 'barbecue',

    # Common name components
    'house', 'grill'
}

CONTEXT_WORDS = {
    # Actions
    'eat', 'ate', 'try', 'tried', 'recommend', 'order',
    'visit', 'went', 'go',

    # Meal types
    'dinner', 'lunch', 'breakfast', 'meal', 'food',

    # Descriptive
    'cuisine', 'menu', 'dish', 'chef', 'reservation',
    'delicious', 'favorite', 'best', 'amazing', 'great',

    # Location words
    'place', 'spot', 'joint'
}

EXCLUSION_WORDS = {
    'i', 'if', 'in', 'it', 'and', 'but', 'what',
    'just', 'made', 'did', 'how', 'why'
}

app = Flask(__name__)

# Set NLTK data path
nltk.data.path.append('/app/nltk_data')

# Print credentials check
logger.info("Checking credentials...")
logger.info(f"Client ID exists: {'REDDIT_CLIENT_ID' in os.environ}")
logger.info(f"Client Secret exists: {'REDDIT_CLIENT_SECRET' in os.environ}")
logger.info(f"Google Maps API Key exists: {'GOOGLE_MAPS_API_KEY' in os.environ}")

reddit = praw.Reddit(
    client_id=os.environ['REDDIT_CLIENT_ID'],
    client_secret=os.environ['REDDIT_CLIENT_SECRET'],
    user_agent="script:restaurant-mapper:v1.0",
    check_for_async=False
)

def extract_nyc_location(text):
    """Extract NYC-specific location context from text."""
    text_lower = text.lower()

    # Check for neighborhood mentions
    for key, value in NYC_NEIGHBORHOODS.items():
        if key in text_lower:
            return value

    # If no specific neighborhood found but NYC is mentioned, default to general NYC
    nyc_indicators = ['nyc', 'new york', 'manhattan', 'brooklyn', 'queens']
    if any(indicator in text_lower for indicator in nyc_indicators):
        return 'New York, NY'

    return None

def extract_location_context(submission):
    """
    Extract location context from Reddit post using multiple methods:
    1. Subreddit name matching
    2. NYC neighborhood detection
    3. Title analysis via Claude
    Returns location in 'City, State' or neighborhood-specific format
    """
    # Check subreddit name for location
    subreddit_name = submission.subreddit.display_name.lower()
    if subreddit_name in CITY_SUBREDDITS:
        if CITY_SUBREDDITS[subreddit_name] == 'New York, NY':
            # For NYC subreddits, try to get more specific neighborhood context
            nyc_location = extract_nyc_location(submission.title)
            return nyc_location if nyc_location else 'New York, NY'
        return CITY_SUBREDDITS[subreddit_name]

    # Try to extract NYC-specific location
    nyc_location = extract_nyc_location(submission.title)
    if nyc_location:
        return nyc_location

    # Fall back to Claude for other locations
    try:
        context_elements = [
            f"Subreddit: r/{submission.subreddit.display_name}",
            f"Title: {submission.title}"
        ]

        if hasattr(submission, 'link_flair_text') and submission.link_flair_text:
            context_elements.append(f"Flair: {submission.link_flair_text}")

        anthropic = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

        prompt = f"""Analyze this Reddit post information and determine its location:

        {chr(10).join(context_elements)}

        For NYC posts, specify the neighborhood if possible (e.g., 'Upper East Side, Manhattan, NY' or 'Williamsburg, Brooklyn, NY').
        For other locations, use 'City, State' format (e.g., 'Boston, MA').
        Return 'None' if location is unclear.
        Do not guess or make assumptions.

        Return only the location or 'None' with no other text."""

        message = anthropic.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            temperature=0,
            system="Extract location from Reddit post context. Return only location format or None.",
            messages=[{"role": "user", "content": prompt}]
        )

        location = message.content[0].text.strip()
        return None if location.lower() == 'none' else location

    except Exception as e:
        logger.error(f"Error extracting location context: {str(e)}")
        return None

def find_restaurants_nlp(text):
    """Extract potential restaurant names from text using NLP techniques."""
    restaurants = set()

    # Split text into sentences for better context
    sentences = text.split('.')

    for sentence in sentences:
        sentence = sentence.strip()
        sentence_lower = sentence.lower()

        # Look for capitalized phrases that might be restaurant names
        words = sentence.split()
        for i in range(len(words)):
            if words[i][0:1].isupper():
                potential_name = words[i]
                j = i + 1
                while j < len(words) and (words[j][0:1].isupper() or words[j].lower() in RESTAURANT_INDICATORS):
                    potential_name += ' ' + words[j]
                    j += 1

                if (len(potential_name.split()) >= 1 and
                    not any(potential_name.lower().startswith(word + ' ') for word in EXCLUSION_WORDS) and
                    len(potential_name) > 2):

                    context_before = ' '.join(words[max(0, i-3):i]).lower()

                    if (any(indicator in potential_name.lower() for indicator in RESTAURANT_INDICATORS) or
                        any(word in context_before.split() for word in CONTEXT_WORDS) or
                        any(word in sentence_lower.split() for word in CONTEXT_WORDS)):
                        restaurants.add(potential_name)

    return list(restaurants)
def construct_search_urls(restaurant_name, location):
    """Create direct search URLs for reservation pages"""
    # Encode the restaurant name and location for URL safety
    search_term = quote(f"{restaurant_name} {location} reservations")

    # Create search URLs for different platforms
    urls = {
        'google_search': f"https://www.google.com/search?q={search_term}",
        'opentable_search': f"https://www.opentable.com/s?term={quote(restaurant_name)}",
        'resy_search': f"https://resy.com/cities/ny/search/{quote(restaurant_name)}", # Assuming NY location
        'tock_search': f"https://www.exploretock.com/search?query={quote(restaurant_name)}"
    }
    return urls

def find_booking_url(restaurant_name, location):
    """Scrape Google search results to find the most likely booking URL"""
    try:
        # Create a search query focused on reservations
        search_query = quote(f"{restaurant_name} {location} reservations book table")
        url = f"https://www.google.com/search?q={search_query}"

        # Use a rotating user agent to avoid blocking
        ua = UserAgent()
        headers = {'User-Agent': ua.random}

        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        # List of common reservation platforms
        booking_domains = [
            'resy.com', 'opentable.com', 'sevenrooms.com', 
            'exploretock.com', 'tablein.com', 'yelp.com/reservations'
        ]

        # First try to find direct booking links
        for link in soup.find_all('a'):
            href = link.get('href')
            if not href:
                continue

            # Clean up the Google redirect URL
            if href.startswith('/url?q='):
                href = href.split('/url?q=')[1].split('&')[0]

            # Check if it's a booking platform URL
            if any(domain in href.lower() for domain in booking_domains):
                return href

        # If no booking platform found, return the restaurant's website
        for link in soup.find_all('a'):
            href = link.get('href')
            if not href:
                continue

            if href.startswith('/url?q='):
                href = href.split('/url?q=')[1].split('&')[0]

            # Skip common non-restaurant websites
            skip_domains = ['google.com', 'facebook.com', 'yelp.com/biz', 
                          'tripadvisor.com', 'wikipedia.org']
            if not any(domain in href.lower() for domain in skip_domains):
                return href

        return None

    except Exception as e:
        logger.error(f"Error finding booking URL: {str(e)}")
        return None

def find_restaurant_address(restaurant_name, location_context):
    """Use Google Places APIs to find the full address and details of a restaurant."""
    try:
        # First use Places Autocomplete API to get place_id
        autocomplete_url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
        search_query = f"{restaurant_name} {location_context}" if location_context else restaurant_name

        autocomplete_params = {
            'input': search_query,
            'types': 'restaurant',  # Restrict to restaurants
            'location': location_context,  # Bias results to this area
            'key': os.environ['GOOGLE_MAPS_API_KEY']
        }

        autocomplete_response = requests.get(autocomplete_url, params=autocomplete_params)
        autocomplete_data = autocomplete_response.json()

        if autocomplete_response.status_code == 200 and autocomplete_data.get('predictions'):
            # Get place_id from first result
            place_id = autocomplete_data['predictions'][0]['place_id']

            # Then use Places Details API to get full information
            details_url = "https://maps.googleapis.com/maps/api/place/details/json"
            details_params = {
                'place_id': place_id,
                'fields': 'name,formatted_address,geometry',  # Specify required fields
                'key': os.environ['GOOGLE_MAPS_API_KEY']
            }

            details_response = requests.get(details_url, params=details_params)
            details_data = details_response.json()

            if details_response.status_code == 200 and details_data.get('result'):
                result = details_data['result']
                booking_url = find_booking_url(restaurant_name, result['formatted_address'])

                # Create the final URL here
                search_query = quote(f"{restaurant_name} {result['formatted_address']} reservations")
                google_search_url = f"https://www.google.com/search?q={search_query}"
                final_url = booking_url if booking_url and booking_url.startswith('http') else google_search_url

                return {
                    'name': restaurant_name,
                    'lat': result['geometry']['location']['lat'],
                    'lng': result['geometry']['location']['lng'],
                    'address': result['formatted_address'],
                    'website': result.get('website', ''),
                    'maps_url': result.get('url', ''),
                    'phone': result.get('formatted_phone_number', ''),
                    'rating': result.get('rating', ''),
                    'price_level': result.get('price_level', ''),
                    'booking_url': booking_url,
                    'final_url': final_url,  # Add this new field
                    'button_text': 'Book Now' if booking_url and booking_url.startswith('http') else 'Find Reservations',
                    'confidence': 'high'
                }

        logger.info(f"No results found for {restaurant_name}")
        return None

    except Exception as e:
        logger.info(f"Error finding address for {restaurant_name}: {str(e)}")
        return None

# [Next function below would be geocode_restaurants]
def geocode_restaurants(restaurants, location_context):
    """Find and geocode restaurant locations using Google Places API."""
    logger.info(f"Starting location search for {len(restaurants)} restaurants in {location_context}")
    locations = []

    for restaurant in restaurants:
        location_data = find_restaurant_address(restaurant, location_context)
        if location_data:
            locations.append(location_data)
            logger.info(f"Found location for {restaurant}: {location_data['address']}")
        else:
            logger.info(f"Could not find location for {restaurant}")

    return locations

# [Next function below would be verify_restaurants_with_claude]

def verify_restaurants_with_claude(text_content, potential_restaurants):
    """Verify restaurant names using Claude AI. Returns a list of confirmed restaurant names."""
    if not potential_restaurants:
        return []

    try:
        anthropic = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

        time.sleep(0.5)  # Add this line - waits 0.5 seconds between calls

        message = anthropic.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=300,
            temperature=0,
            system="Extract restaurant names from text. Return only valid JSON arrays.",
            messages=[{
                "role": "user", 
                "content": f"""Here's text about restaurants:
                {text_content}

                Potential restaurant names:
                {json.dumps(potential_restaurants)}

                Return only the items that are actually restaurant names mentioned in the text, as a JSON array. Exclude any awards, chef names, cities, descriptive phrases, or non-restaurant items."""
            }]
        )

        response_text = message.content[0].text
        start = response_text.find('[')
        end = response_text.rfind(']') + 1

        return json.loads(response_text[start:end]) if start != -1 and end != 0 else []

    except Exception as e:
        logger.info(f"Error in Claude verification: {str(e)}")
        return []  # Return empty list instead of falling back

def extract_post_id_from_url(url):
    pattern = r'reddit\.com/r/\w+/comments/(\w+)(?:/[^?]*)?'
    match = re.search(pattern, url)
    if match:
        post_id = match.group(1)
        logger.info(f"Extracted post ID: {post_id}")
        return post_id
    logger.info("Failed to extract post ID")
    return None

def process_text_for_restaurants(text, text_type="text"):
    """Process a piece of text to find and verify restaurants.
    Returns tuple of (verified_restaurants, text_details if restaurants found else None)"""
    logger.info(f"Processing {text_type}: {text[:100]}...")
    potential_restaurants = find_restaurants_nlp(text)

    if potential_restaurants:
        verified_restaurants = verify_restaurants_with_claude(text, potential_restaurants)
        if verified_restaurants:
            logger.info(f"Found verified restaurants in {text_type}: {verified_restaurants}")
            return verified_restaurants, {
                'text': text,
                'restaurants': verified_restaurants
            }
    return [], None

def scan_post(url, upvote_threshold=5, top_level_only=False):
    logger.info(f"\nAttempting to scan URL: {url}")

    post_id = extract_post_id_from_url(url)
    if not post_id:
        logger.info("Invalid URL format")
        return "Invalid URL format"

    try:
        logger.info(f"Attempting to fetch submission with ID: {post_id}")
        submission = reddit.submission(id=post_id)

        logger.info("Successfully got submission")
        logger.info(f"Title type: {type(submission.title)}")
        logger.info(f"Title content: {submission.title}")

        # Get location context early
        location_context = extract_location_context(submission)
        logger.info(f"Detected location context: {location_context}")

        results = {
            'title': submission.title,
            'restaurants': {},
            'comments_with_restaurants': []
        }

        logger.info("Attempting to load comments...")
        submission.comments.replace_more(limit=None)

        # Get comments based on user preference
        submission.comments.replace_more(limit=2)  # Reduce from None to 2
        if top_level_only:
            comments = submission.comments._comments[:75]  # Limit to 50 comments
            logger.info(f"Processing up to 20 top-level comments")
        else:
            comments = submission.comments.list()[:75]  # Limit to 50 total comments
            logger.info(f"Processing up to 50 total comments")

        # Process comments
        filtered_comments = [comment for comment in comments if hasattr(comment, 'score') and comment.score >= upvote_threshold]
        logger.info(f"Filtered down to {len(filtered_comments)} comments with {upvote_threshold}+ upvotes")

        # Create a dictionary to track restaurant metrics
        restaurant_metrics = {}

        # Process title first
        title_restaurants, _ = process_text_for_restaurants(submission.title, "title")
        if title_restaurants:
            for restaurant in title_restaurants:
                if restaurant not in restaurant_metrics:
                    restaurant_metrics[restaurant] = {
                        'comment_count': 1,
                        'total_upvotes': 0,  # Title doesn't have upvotes
                        'comments': []
                    }

        # Process comments
        for comment in filtered_comments:
            verified_restaurants, comment_details = process_text_for_restaurants(comment.body, "comment")
            if comment_details:
                results['comments_with_restaurants'].append(comment_details)
                # Track metrics for each restaurant
                for restaurant in verified_restaurants:
                    if restaurant not in restaurant_metrics:
                        restaurant_metrics[restaurant] = {
                            'comment_count': 0,
                            'total_upvotes': 0,
                            'comments': []
                        }
                    restaurant_metrics[restaurant]['comment_count'] += 1
                    restaurant_metrics[restaurant]['total_upvotes'] += comment.score
                    restaurant_metrics[restaurant]['comments'].append({
                        'text': comment.body,
                        'upvotes': comment.score
                    })

        # Sort restaurants by metrics
        sorted_restaurants = sorted(
            restaurant_metrics.items(),
            key=lambda x: (x[1]['comment_count'], x[1]['total_upvotes']),
            reverse=True
        )

        # Get locations with metrics
        locations = []
        for restaurant, metrics in sorted_restaurants:
            location_data = find_restaurant_address(restaurant, location_context)
            if location_data:
                location_data.update({
                    'comment_count': metrics['comment_count'],
                    'total_upvotes': metrics['total_upvotes']
                })
                locations.append(location_data)
                logger.info(f"Found location for {restaurant} with {metrics['comment_count']} comments and {metrics['total_upvotes']} upvotes")

        results['locations'] = locations

        logger.info("Final results structure:", {
            'title': results['title'],
            'restaurant_count': len(restaurant_metrics),
            'location_count': len(results.get('locations', [])),
            'comment_count': len(results['comments_with_restaurants'])
        })

        return results

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        return f"Error: {e}"

# In main.py, modify the Flask route:
@app.route('/', methods=['GET', 'POST'])
def home():
    results = None
    if request.method == 'POST':
        url = request.form.get('url')
        threshold = request.form.get('threshold')
        top_level_only = request.form.get('top_level_only') == 'true'

        try:
            # Add timeout handling
            from werkzeug.serving import is_running_from_reloader
            if not is_running_from_reloader():
                logger.info("Starting request with timeout...")
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError("Request took too long to process")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(25)  # Set 25-second timeout

            results = scan_post(url, threshold, top_level_only)

            if not is_running_from_reloader():
                signal.alarm(0)  # Disable alarm

        except TimeoutError:
            logger.error("Request timed out")
            return render_template('index.html', 
                                error="Request timed out. Please try again with fewer comments or top-level comments only.",
                                google_maps_api_key=os.environ['GOOGLE_MAPS_API_KEY'])
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return render_template('index.html',
                                error=f"Error processing request: {str(e)}",
                                google_maps_api_key=os.environ['GOOGLE_MAPS_API_KEY'])

    return render_template('index.html', 
                         results=results, 
                         google_maps_api_key=os.environ['GOOGLE_MAPS_API_KEY'])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)