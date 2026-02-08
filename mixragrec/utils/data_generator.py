"""
Data generator for creating synthetic user queries and contexts for training and testing.

Part of MixRAGRec framework.
"""

import random
import numpy as np
from typing import Dict, List, Any, Optional
import json


class DataGenerator:
    """"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.seed = config.get('seed', 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        self.query_templates = [
            "I need recommendations for {product_type}",
            "Can you suggest some {product_type} for {user_context}?",
            "What are the best {product_type} available?",
            "I'm looking for {product_type} that {requirement}",
            "Help me find {product_type} suitable for {occasion}",
            "Recommend {product_type} within {budget} budget",
            "What {product_type} would you suggest for {user_preference}?",
            "I want to buy {product_type} for {purpose}",
            "Show me {product_type} that are {quality_aspect}",
            "Which {product_type} are popular among {user_group}?"
        ]
        
        self.product_types = [
            "books", "movies", "restaurants", "laptops", "smartphones",
            "headphones", "games", "courses", "hotels", "travel destinations",
            "clothing", "shoes", "furniture", "appliances", "cars",
            "software", "apps", "music", "podcasts", "recipes"
        ]
        
        self.user_contexts = [
            "work", "home", "travel", "gift giving", "personal use",
            "professional development", "entertainment", "fitness",
            "cooking", "studying", "relaxation", "outdoor activities"
        ]
        
        self.requirements = [
            "are affordable", "have good reviews", "are highly rated",
            "are popular", "are innovative", "are user-friendly",
            "have advanced features", "are reliable", "are stylish",
            "are eco-friendly", "are portable", "are durable"
        ]
        
        self.occasions = [
            "business meetings", "casual use", "special events",
            "daily commute", "weekend activities", "family gatherings",
            "outdoor adventures", "indoor activities", "formal occasions"
        ]
        
        self.budgets = [
            "low", "moderate", "high", "unlimited", "under $100",
            "under $500", "under $1000", "premium", "budget-friendly"
        ]
        
        self.user_preferences = [
            "beginners", "experts", "professionals", "students",
            "families", "individuals", "tech enthusiasts", "casual users",
            "power users", "budget-conscious buyers", "quality seekers"
        ]
        
        self.purposes = [
            "personal enjoyment", "professional work", "learning",
            "entertainment", "productivity", "communication",
            "creativity", "fitness", "health", "convenience"
        ]
        
        self.quality_aspects = [
            "high quality", "good value", "well-designed", "feature-rich",
            "easy to use", "highly recommended", "award-winning",
            "cutting-edge", "reliable", "versatile"
        ]
        
        self.user_groups = [
            "young adults", "professionals", "students", "families",
            "seniors", "teenagers", "experts", "beginners",
            "tech users", "casual users"
        ]
        
        self.user_types = {
            'tech_enthusiast': {
                'preferences': ['latest technology', 'advanced features', 'high performance'],
                'budget_tendency': 'high',
                'product_interests': ['laptops', 'smartphones', 'software', 'apps']
            },
            'budget_conscious': {
                'preferences': ['good value', 'affordable', 'cost-effective'],
                'budget_tendency': 'low',
                'product_interests': ['books', 'clothing', 'apps', 'courses']
            },
            'quality_seeker': {
                'preferences': ['premium quality', 'highly rated', 'well-reviewed'],
                'budget_tendency': 'high',
                'product_interests': ['restaurants', 'hotels', 'furniture', 'cars']
            },
            'casual_user': {
                'preferences': ['easy to use', 'simple', 'convenient'],
                'budget_tendency': 'moderate',
                'product_interests': ['movies', 'music', 'games', 'recipes']
            },
            'professional': {
                'preferences': ['reliable', 'professional grade', 'efficient'],
                'budget_tendency': 'moderate',
                'product_interests': ['laptops', 'software', 'courses', 'books']
            }
        }
        
        self.sample_documents = self._create_sample_documents()
        
    def generate_user_query(self) -> str:
        """"""
        template = random.choice(self.query_templates)
        
        query = template.format(
            product_type=random.choice(self.product_types),
            user_context=random.choice(self.user_contexts),
            requirement=random.choice(self.requirements),
            occasion=random.choice(self.occasions),
            budget=random.choice(self.budgets),
            user_preference=random.choice(self.user_preferences),
            purpose=random.choice(self.purposes),
            quality_aspect=random.choice(self.quality_aspects),
            user_group=random.choice(self.user_groups)
        )
        
        return query
    
    def generate_user_context(self) -> Dict[str, Any]:
        """"""
        user_type = random.choice(list(self.user_types.keys()))
        user_profile = self.user_types[user_type]
        
        context = {
            'user_type': user_type,
            'preferences': random.choice(user_profile['preferences']),
            'budget_tendency': user_profile['budget_tendency'],
            'product_interests': user_profile['product_interests'],
            'preference_score': random.uniform(0.3, 0.9),
            'session_length': random.randint(1, 5),
            'previous_satisfaction': random.uniform(0.2, 0.8),
            'time_of_day': random.randint(6, 23),
            'device_type': random.choice(['mobile', 'desktop', 'tablet']),
            'location': random.choice(['home', 'office', 'public', 'travel'])
        }
        
        return context
    
    def generate_follow_up_query(self, 
                                original_query: str,
                                previous_response: str,
                                user_context: Dict[str, Any]) -> str:
        """"""
        
        follow_up_templates = [
            "Can you provide more details about {aspect}?",
            "What about alternatives to {mention}?",
            "Are there any {context} options?",
            "Can you explain why you recommended {mention}?",
            "What are the pros and cons of {mention}?",
            "Do you have recommendations in a different price range?",
            "What if I prefer {preference} instead?",
            "Are there any newer options available?",
            "Can you suggest something more specific?",
            "What would you recommend for a different {context}?"
        ]
        
        aspects = ['features', 'price', 'quality', 'usability', 'design']
        mentions = self._extract_mentions(previous_response)
        contexts = ['budget-friendly', 'premium', 'portable', 'professional']
        preferences = user_context.get('preferences', 'quality')
        
        template = random.choice(follow_up_templates)
        follow_up = template.format(
            aspect=random.choice(aspects),
            mention=random.choice(mentions) if mentions else 'that',
            context=random.choice(contexts),
            preference=preferences
        )
        
        return follow_up
    
    def _extract_mentions(self, text: str) -> List[str]:
        """"""
        words = text.split()
        mentions = []
        
        for word in words:
            if len(word) > 4 and word.lower() not in ['this', 'that', 'these', 'those', 'with', 'from']:
                mentions.append(word)
        
        return mentions[:3]
    
    def _create_sample_documents(self) -> List[str]:
        """"""
        documents = []
        
        for product_type in self.product_types:
            for i in range(5):
                doc = self._generate_product_document(product_type, i)
                documents.append(doc)
        
        return documents
    
    def _generate_product_document(self, product_type: str, index: int) -> str:
        """"""
        
        doc_templates = [
            "The {product} is a {quality} {product_type} that offers {features}. "
            "It's perfect for {use_case} and has received {rating} from users. "
            "Key benefits include {benefits}. Price range: {price_range}.",
            
            "{product} stands out as a {quality} option in the {product_type} category. "
            "Users appreciate its {features} and {benefits}. "
            "Ideal for {use_case} with a {rating} rating. {price_range}.",
            
            "For those seeking {quality} {product_type}, {product} delivers {features}. "
            "This {product_type} excels in {use_case} scenarios. "
            "Customer feedback: {rating}. {benefits}. {price_range}."
        ]
        
        product_names = {
            'books': ['The Complete Guide', 'Essential Handbook', 'Masterclass Series'],
            'laptops': ['TechPro X1', 'PowerBook Elite', 'UltraSlim Pro'],
            'smartphones': ['PhoneMax Pro', 'SmartEdge Plus', 'TechPhone Ultra'],
            'restaurants': ['Gourmet Bistro', 'Casual Corner', 'Fine Dining Plus']
        }
        
        product_name = random.choice(product_names.get(product_type, [f'Premium {product_type.title()}']))
        
        qualities = ['excellent', 'outstanding', 'remarkable', 'superior', 'exceptional']
        features = ['advanced functionality', 'user-friendly design', 'innovative features', 'robust performance']
        use_cases = ['professional work', 'daily use', 'special occasions', 'casual activities']
        ratings = ['excellent reviews', '5-star ratings', 'high customer satisfaction', 'positive feedback']
        benefits = ['durability', 'value for money', 'ease of use', 'versatility']
        price_ranges = ['affordable pricing', 'mid-range cost', 'premium pricing', 'budget-friendly']
        
        template = random.choice(doc_templates)
        document = template.format(
            product=product_name,
            product_type=product_type,
            quality=random.choice(qualities),
            features=random.choice(features),
            use_case=random.choice(use_cases),
            rating=random.choice(ratings),
            benefits=random.choice(benefits),
            price_range=random.choice(price_ranges)
        )
        
        return document
    
    def get_sample_documents(self, sample_size: Optional[int] = None) -> List[str]:
        """"""
        if sample_size is None:
            return self.sample_documents
        else:
            return random.sample(self.sample_documents, min(sample_size, len(self.sample_documents)))
    
    def generate_training_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """"""
        batch = []
        
        for _ in range(batch_size):
            item = {
                'user_query': self.generate_user_query(),
                'user_context': self.generate_user_context(),
                'conversation_history': []
            }
            batch.append(item)
        
        return batch
    
    def create_evaluation_dataset(self, size: int) -> List[Dict[str, Any]]:
        """"""
        dataset = []
        
        user_types = list(self.user_types.keys())
        
        for i in range(size):
            user_type = user_types[i % len(user_types)]
            
            user_profile = self.user_types[user_type]
            context = {
                'user_type': user_type,
                'preferences': random.choice(user_profile['preferences']),
                'budget_tendency': user_profile['budget_tendency'],
                'product_interests': user_profile['product_interests'],
                'preference_score': random.uniform(0.5, 0.9),
                'session_length': 1,
                'previous_satisfaction': 0.5,
                'time_of_day': 12,
                'device_type': 'desktop',
                'location': 'home'
            }
            
            product_type = random.choice(user_profile['product_interests'])
            query_template = random.choice(self.query_templates)
            query = query_template.format(
                product_type=product_type,
                user_context=random.choice(self.user_contexts),
                requirement=random.choice(self.requirements),
                occasion=random.choice(self.occasions),
                budget=user_profile['budget_tendency'],
                user_preference=user_profile['preferences'][0],
                purpose=random.choice(self.purposes),
                quality_aspect=random.choice(self.quality_aspects),
                user_group=random.choice(self.user_groups)
            )
            
            item = {
                'id': f'eval_{i}',
                'user_query': query,
                'user_context': context,
                'conversation_history': [],
                'expected_product_type': product_type,
                'expected_user_type': user_type
            }
            
            dataset.append(item)
        
        return dataset
