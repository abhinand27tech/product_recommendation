import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import fpgrowth, apriori
from mlxtend.frequent_patterns import association_rules
from collections import defaultdict
from scipy.sparse import csr_matrix
import os
import traceback

class RecommendationSystem:
    # Initializes variables needed across methods
    def __init__(self):
        self.transactions_df = None
        self.customer_df = None
        self.user_item_matrix = None
        self.item_item_similarity = None
        self.user_user_similarity = None
        self.rules = None

    def load_data(self, detail_path, header_path, progress_callback=None):
        """Load and preprocess the transaction data"""
        try:
            if progress_callback:
                progress_callback("Loading header file...")
            
            if not os.path.exists(detail_path):
                raise FileNotFoundError(f"Detail file not found: {detail_path}")
            if not os.path.exists(header_path):
                raise FileNotFoundError(f"Header file not found: {header_path}")
            
            # Load header file first (usually smaller)
            if progress_callback:
                progress_callback("Reading header file...")
            
            header_df = pd.read_csv(header_path, 
                                  usecols=['voucher_id', 'cust_code', 'godown_code', 'register_code'],
                                  dtype={
                                      'voucher_id': str,
                                      'cust_code': str,
                                      'godown_code': str,
                                      'register_code': str
                                  })
            
            # Create unique IDs for anonymous customers (cust_code = '0')
            # by combining with godown_code
            header_df['compound_cust_code'] = header_df.apply(
                lambda row: f"0_{row['godown_code']}" if row['cust_code'] == '0' else row['cust_code'],
                axis=1
            )
            
            if progress_callback:
                progress_callback(f"Analyzing {len(header_df)} header records...")
            
            # Debug logging for godown analysis
            print("\nInitial godown analysis:")
            print(f"Total unique godowns in header: {len(header_df['godown_code'].unique())}")
            print(f"All godown codes: {sorted(header_df['godown_code'].unique())}")
            print("\nRegister code distribution:")
            reg_codes = header_df['register_code'].value_counts()
            for code, count in reg_codes.items():
                print(f"Register code '{code}': {count} records")
            
            # Store all customer data for godown listing
            self.customer_df = header_df.copy()
            
            # Filter for register_code = 7 (valid transactions)
            valid_headers = header_df[header_df['register_code'] == '7'].copy()
            
            # Debug logging for valid transactions
            print("\nValid transaction analysis:")
            print(f"Valid transactions: {len(valid_headers)}")
            print(f"Valid unique godowns: {len(valid_headers['godown_code'].unique())}")
            print(f"Valid godown codes: {sorted(valid_headers['godown_code'].unique())}")
            
            if progress_callback:
                progress_callback(f"Found {len(valid_headers)} valid transactions...")
            
            # Get valid voucher IDs for filtering detail records
            valid_vouchers = set(valid_headers['voucher_id'])
            
            if progress_callback:
                progress_callback("Processing detail file in chunks...")
            
            chunk_size = 50000  # Reduced chunk size
            chunks = []
            total_rows = 0
            
            # Process detail file in chunks
            for chunk_num, chunk in enumerate(pd.read_csv(detail_path, 
                                   chunksize=chunk_size,
                                   usecols=['voucher_id', 'item_no'],  # Only load needed columns
                                   dtype={'voucher_id': str, 'item_no': str})):
                # Filter chunk to only include valid vouchers
                filtered_chunk = chunk[chunk['voucher_id'].isin(valid_vouchers)]
                if len(filtered_chunk) > 0:
                    chunks.append(filtered_chunk)
                    total_rows += len(filtered_chunk)
                    if progress_callback:
                        progress_callback(f"Processed chunk {chunk_num + 1}: {total_rows} valid records so far")
            
                # Process in smaller batches to manage memory
                if len(chunks) >= 20:  # Process every 20 chunks
                    if progress_callback:
                        progress_callback("Merging intermediate chunks to save memory...")
                    intermediate_df = pd.concat(chunks, ignore_index=True)
                    chunks = [intermediate_df]
            
            if progress_callback:
                progress_callback("Merging final chunks...")
            
            if chunks:
                detail_df = pd.concat(chunks, ignore_index=True)
                if progress_callback:
                    progress_callback(f"Total valid detail records: {len(detail_df)}")
            else:
                raise ValueError("No valid detail records found after filtering")
            
            if progress_callback:
                progress_callback("Merging with header data...")
            
            # Merge datasets using valid transactions only
            merge_columns = ['voucher_id', 'cust_code', 'compound_cust_code', 'godown_code', 'register_code']
            self.transactions_df = pd.merge(
                detail_df,
                valid_headers[merge_columns],
                on='voucher_id',
                how='inner'
            )
            
            # Clear memory
            del detail_df
            del valid_headers
            del chunks
            
            if progress_callback:
                progress_callback("Creating user-item matrix...")
            
            # Create user-item matrix
            if not self.create_user_item_matrix():
                raise Exception("Failed to create user-item matrix")
            
            if progress_callback:
                progress_callback("Computing similarity matrices...")
            
            # Compute similarity matrices
            if not self.compute_similarity_matrices():
                raise Exception("Failed to compute similarity matrices")
            
            if progress_callback:
                progress_callback("Generating association rules...")
            
            # Generate association rules
            self.generate_rules()
            
            if progress_callback:
                progress_callback("Data loading and preprocessing completed successfully")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            traceback.print_exc()  # Print the full error traceback
            return False

    def create_user_item_matrix(self):
        """Create user-item purchase matrix with memory optimization"""
        try:
            # Create sparse matrix for efficiency
            customers = self.transactions_df['compound_cust_code'].unique()
            items = self.transactions_df['item_no'].unique()
            
            # Create mappings for efficient matrix creation
            customer_to_idx = {cust: idx for idx, cust in enumerate(customers)}
            item_to_idx = {item: idx for idx, item in enumerate(items)}
            
            # Create sparse matrix data
            rows = [customer_to_idx[cust] for cust in self.transactions_df['compound_cust_code']]
            cols = [item_to_idx[item] for item in self.transactions_df['item_no']]
            data = np.ones(len(rows))  # Binary purchase data
            
            # Create sparse matrix
            self.user_item_matrix = csr_matrix(
                (data, (rows, cols)), 
                shape=(len(customers), len(items))
            )
            
            # Store mappings for later use
            self.customer_mapping = {idx: cust for cust, idx in customer_to_idx.items()}
            self.item_mapping = {idx: item for item, idx in item_to_idx.items()}
            
            print(f"Created user-item matrix with shape: {self.user_item_matrix.shape}")
            return True
        except Exception as e:
            print(f"Error creating user-item matrix: {str(e)}")
            return False

    def compute_similarity_matrices(self):
        """Compute both user-user and item-item similarity matrices"""
        try:
            print("Computing similarity matrices...")
            
            # User-User similarity
            user_similarity = cosine_similarity(self.user_item_matrix)
            self.user_user_similarity = pd.DataFrame(
                user_similarity,
                index=[self.customer_mapping[i] for i in range(len(self.customer_mapping))],
                columns=[self.customer_mapping[i] for i in range(len(self.customer_mapping))]
            )
            
            # Item-Item similarity
            item_similarity = cosine_similarity(self.user_item_matrix.T)
            self.item_item_similarity = pd.DataFrame(
                item_similarity,
                index=[self.item_mapping[i] for i in range(len(self.item_mapping))],
                columns=[self.item_mapping[i] for i in range(len(self.item_mapping))]
            )
            
            print("Similarity matrices computed successfully")
            return True
        except Exception as e:
            print(f"Error computing similarity matrices: {str(e)}")
            return False

    def generate_rules(self):
        """Generate association rules using memory-efficient approach"""
        try:
            # Sample transactions more aggressively
            if len(self.transactions_df) > 1000000:
                sampled_df = self.transactions_df.sample(n=10000, random_state=42)
            else:
                sampled_df = self.transactions_df.sample(n=min(10000, len(self.transactions_df)), random_state=42)

            print(f"Processing {len(sampled_df)} transactions for rules generation...")
            
            # Create transaction list format (more memory efficient)
            transactions = sampled_df.groupby('voucher_id')['item_no'].agg(list)
            
            # Convert to format suitable for efficient Apriori
            records = []
            for items in transactions:
                records.append(frozenset(items))
            
            print(f"Created {len(records)} transaction records")
            
            try:
                # Step 1: Find frequent individual items (lowered threshold)
                min_support_count = max(2, int(len(records) * 0.0005))  # At least 2 occurrences or 0.05%
                item_counts = defaultdict(int)
                for transaction in records:
                    for item in transaction:
                        item_counts[item] += 1
                
                # Filter for frequent items
                frequent_items = {item: count for item, count in item_counts.items() 
                                if count >= min_support_count}
                
                print(f"Found {len(frequent_items)} frequent individual items")
                
                # Step 2: Generate pairs only for frequent items
                pair_counts = defaultdict(int)
                for transaction in records:
                    # Convert to set for O(1) lookup
                    transaction_set = set(item for item in transaction if item in frequent_items)
                    # Generate pairs
                    items_list = sorted(transaction_set)  # Sort to ensure consistent ordering
                    for i in range(len(items_list)):
                        for j in range(i + 1, len(items_list)):
                            pair_counts[(items_list[i], items_list[j])] += 1
                
                # Filter for frequent pairs (lowered threshold)
                frequent_pairs = {pair: count for pair, count in pair_counts.items() 
                                if count >= min_support_count}
                
                print(f"Found {len(frequent_pairs)} frequent pairs")
                
                # Step 3: Generate rules from frequent pairs
                rules_data = []
                total_transactions = len(records)
                
                for (item1, item2), pair_count in frequent_pairs.items():
                    # Calculate confidence and lift
                    support_pair = pair_count / total_transactions
                    support_item1 = item_counts[item1] / total_transactions
                    support_item2 = item_counts[item2] / total_transactions
                    
                    # Generate rule item1 -> item2 (lowered confidence threshold)
                    confidence_1_2 = pair_count / item_counts[item1]
                    lift_1_2 = confidence_1_2 / support_item2
                    
                    if confidence_1_2 >= 0.005:  # 0.5% confidence threshold
                        rules_data.append({
                            'antecedents': frozenset([item1]),
                            'consequents': frozenset([item2]),
                            'support': support_pair,
                            'confidence': confidence_1_2,
                            'lift': lift_1_2
                        })
                    
                    # Generate rule item2 -> item1
                    confidence_2_1 = pair_count / item_counts[item2]
                    lift_2_1 = confidence_2_1 / support_item1
                    
                    if confidence_2_1 >= 0.005:  # 0.5% confidence threshold
                        rules_data.append({
                            'antecedents': frozenset([item2]),
                            'consequents': frozenset([item1]),
                            'support': support_pair,
                            'confidence': confidence_2_1,
                            'lift': lift_2_1
                        })
                
                # Convert to DataFrame and sort by lift
                self.rules = pd.DataFrame(rules_data)
                if not self.rules.empty:
                    self.rules = self.rules.nlargest(5000, columns=['lift'])
                    print(f"Successfully generated {len(self.rules)} rules")
                else:
                    print("Warning: No rules generated with current thresholds")
                    # Fallback to simple frequency-based recommendations
                    top_items = sorted([(count, item) for item, count in item_counts.items()], reverse=True)[:100]
                    rules_data = []
                    for _, item in top_items:
                        rules_data.append({
                            'antecedents': frozenset(['FREQUENT']),
                            'consequents': frozenset([item]),
                            'support': item_counts[item] / total_transactions,
                            'confidence': 1.0,
                            'lift': 1.0
                        })
                    self.rules = pd.DataFrame(rules_data)
                    print("Generated fallback rules based on item frequency")
                
            except Exception as e:
                print(f"Error generating rules: {str(e)}")
                self.rules = pd.DataFrame()
            
        except Exception as e:
            print(f"Error in rule generation: {str(e)}")
            self.rules = pd.DataFrame()

    def get_user_based_recommendations(self, customer_id, n_recommendations=5, godown_code=None):
        """Get recommendations based on similar users"""
        try:
            # For anonymous customers (ID '0'), convert to compound ID
            if customer_id == '0' and godown_code:
                compound_id = f"0_{godown_code}"
            else:
                # For regular customers, first check if we have a compound ID
                matching_rows = self.transactions_df[self.transactions_df['cust_code'] == customer_id]
                if len(matching_rows) > 0:
                    # Use the first compound ID found for this customer
                    compound_id = matching_rows['compound_cust_code'].iloc[0]
                else:
                    # If customer not found, use as is (will likely return [])
                    compound_id = customer_id
            
            if compound_id not in self.user_user_similarity.index:
                return []
            
            # Get similar users
            similar_users = self.user_user_similarity[compound_id].sort_values(ascending=False)[1:11]
            
            # Get items bought by this customer
            customer_items = set(self.transactions_df[
                self.transactions_df['compound_cust_code'] == compound_id
            ]['item_no'])
            
            # Calculate item scores based on similar users
            item_scores = defaultdict(float)
            for user, similarity in similar_users.items():
                user_items = set(self.transactions_df[
                    self.transactions_df['compound_cust_code'] == user
                ]['item_no'])
                
                for item in user_items:
                    if item not in customer_items:
                        item_scores[item] += similarity
            
            # Sort and return top recommendations
            recommendations = sorted(
                item_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:n_recommendations]
            
            return [item for item, score in recommendations]
        except Exception as e:
            print(f"Error in user-based recommendations: {str(e)}")
            return []

    def get_item_based_recommendations(self, customer_id, n_recommendations=5, godown_code=None):
        """Get recommendations based on similar items"""
        try:
            # For anonymous customers (ID '0'), convert to compound ID
            if customer_id == '0' and godown_code:
                compound_id = f"0_{godown_code}"
            else:
                # For regular customers, first check if we have a compound ID
                matching_rows = self.transactions_df[self.transactions_df['cust_code'] == customer_id]
                if len(matching_rows) > 0:
                    # Use the first compound ID found for this customer
                    compound_id = matching_rows['compound_cust_code'].iloc[0]
                else:
                    # If customer not found, use as is (will likely return [])
                    compound_id = customer_id
            
            # Get customer's previous purchases
            customer_items = set(self.transactions_df[
                self.transactions_df['compound_cust_code'] == compound_id
            ]['item_no'])
            
            if not customer_items:
                return []
            
            # Calculate item scores based on similarity to purchased items
            item_scores = defaultdict(float)
            for bought_item in customer_items:
                if bought_item in self.item_item_similarity.index:
                    similar_items = self.item_item_similarity[bought_item].sort_values(ascending=False)[1:11]
                    for item, similarity in similar_items.items():
                        if item not in customer_items:
                            item_scores[item] += similarity
            
            # Sort and return top recommendations
            recommendations = sorted(
                item_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:n_recommendations]
            
            return [item for item, score in recommendations]
        except Exception as e:
            print(f"Error in item-based recommendations: {str(e)}")
            return []

    def get_association_recommendations(self, customer_id, n_recommendations=5, godown_code=None):
        """Get recommendations based on association rules"""
        if self.rules.empty:
            return []
        
        try:
            # For anonymous customers (ID '0'), convert to compound ID
            if customer_id == '0' and godown_code:
                compound_id = f"0_{godown_code}"
            else:
                # For regular customers, first check if we have a compound ID
                matching_rows = self.transactions_df[self.transactions_df['cust_code'] == customer_id]
                if len(matching_rows) > 0:
                    # Use the first compound ID found for this customer
                    compound_id = matching_rows['compound_cust_code'].iloc[0]
                else:
                    # If customer not found, use as is (will likely return [])
                    compound_id = customer_id
            
            # Get customer's previous purchases
            customer_items = set(self.transactions_df[
                self.transactions_df['compound_cust_code'] == compound_id
            ]['item_no'].unique())
            
            if not customer_items:
                # If no purchase history, return top items by lift
                return list(self.rules.nlargest(n_recommendations, 'lift')['consequents'].apply(list).sum())
            
            # Filter rules based on customer's purchases
            relevant_rules = self.rules[
                self.rules['antecedents'].apply(lambda x: 'FREQUENT' in x or any(item in customer_items for item in x))
            ]
            
            if relevant_rules.empty:
                # If no relevant rules, return top items by lift
                return list(self.rules.nlargest(n_recommendations, 'lift')['consequents'].apply(list).sum())
            
            # Sort by confidence and lift
            relevant_rules = relevant_rules.sort_values(['confidence', 'lift'], ascending=[False, False])
            
            # Get recommended items
            recommendations = []
            seen_items = set()
            for _, rule in relevant_rules.iterrows():
                consequents = list(rule['consequents'])
                for item in consequents:
                    if item not in customer_items and item not in seen_items:
                        recommendations.append(item)
                        seen_items.add(item)
                        if len(recommendations) >= n_recommendations:
                            return recommendations
                        
            return recommendations
            
        except Exception as e:
            print(f"Error in association recommendations: {str(e)}")
            return []

    def validate_customer(self, godown_code, customer_id):
        """Validate if customer belongs to the given godown"""
        try:
            # Check if customer exists in the given godown
            customer_exists = any(
                (self.customer_df['godown_code'] == godown_code) & 
                (self.customer_df['cust_code'] == customer_id)
            )
            return customer_exists
        except Exception as e:
            print(f"Error validating customer: {str(e)}")
            return False

    def get_recommendations(self, customer_id, n_recommendations=10, godown_code=None):
        """Get recommendations for a specific customer with register code 7"""
        try:
            print(f"\nGenerating recommendations for customer {customer_id} in godown {godown_code}...")
            
            # For anonymous customers, construct the compound ID for lookup
            if customer_id == '0' and godown_code:
                compound_id = f"0_{godown_code}"
                # Check if compound ID exists in transactions
                if compound_id not in self.transactions_df['compound_cust_code'].unique():
                    print(f"Anonymous customer in godown {godown_code} has no transactions")
                    return []
            else:
                # For regular customers, check if they exist in transactions
                if customer_id not in self.transactions_df['cust_code'].unique():
                    print("Customer has no transactions")
                    return []
                
                # Try to find the compound ID for this customer
                matching_rows = self.transactions_df[self.transactions_df['cust_code'] == customer_id]
                if len(matching_rows) > 0:
                    compound_id = matching_rows['compound_cust_code'].iloc[0]
                else:
                    compound_id = customer_id
            
            # Get all transactions for this customer (using appropriate ID)
            if customer_id == '0' and godown_code:
                # For anonymous customers, use compound ID
                customer_transactions = self.transactions_df[
                    self.transactions_df['compound_cust_code'] == compound_id
                ]
            else:
                # For regular customers, use regular ID
                customer_transactions = self.transactions_df[
                    self.transactions_df['cust_code'] == customer_id
                ]
            
            print(f"Found {len(customer_transactions)} total transactions")
            
            # Filter for valid transactions (register code 7)
            valid_transactions = customer_transactions[
                customer_transactions['register_code'] == '7'
            ]
            
            print(f"Found {len(valid_transactions)} valid transactions (register code 7)")
            
            if len(valid_transactions) == 0:
                print("No valid transactions found for this customer")
                # Try getting recommendations based on all transactions
                print("Attempting to generate recommendations based on all transactions...")
                
                # Get recommendations from each approach
                user_based_recs = self.get_user_based_recommendations(customer_id, n_recommendations * 2, godown_code)
                item_based_recs = self.get_item_based_recommendations(customer_id, n_recommendations * 2, godown_code)
                assoc_recs = self.get_association_recommendations(customer_id, n_recommendations * 2, godown_code)
                
                # Combine all recommendations
                all_recs = []
                all_recs.extend(user_based_recs[:n_recommendations])
                all_recs.extend(item_based_recs[:n_recommendations])
                all_recs.extend(assoc_recs[:n_recommendations])
                
                # Remove duplicates while preserving order
                seen = set()
                unique_recs = [x for x in all_recs if not (x in seen or seen.add(x))]
                
                return unique_recs[:n_recommendations]
                
            # Get recommendations from each approach
            user_based_recs = self.get_user_based_recommendations(customer_id, n_recommendations * 2, godown_code)
            item_based_recs = self.get_item_based_recommendations(customer_id, n_recommendations * 2, godown_code)
            assoc_recs = self.get_association_recommendations(customer_id, n_recommendations * 2, godown_code)
            
            print(f"Generated recommendations:")
            print(f"User-based: {len(user_based_recs)}")
            print(f"Item-based: {len(item_based_recs)}")
            print(f"Association rules: {len(assoc_recs)}")
            
            # Combine and score recommendations
            rec_scores = defaultdict(float)
            
            # Weight for each method
            weights = {
                'user_based': 0.35,
                'item_based': 0.35,
                'association': 0.30
            }
            
            # Score recommendations from each method
            for item in user_based_recs:
                rec_scores[item] += weights['user_based']
            for item in item_based_recs:
                rec_scores[item] += weights['item_based']
            for item in assoc_recs:
                rec_scores[item] += weights['association']
            
            # Sort by score and return top recommendations
            final_recommendations = sorted(
                rec_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_recommendations]
            
            return [item for item, score in final_recommendations]
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            traceback.print_exc()
            return []

    def get_customers_by_godown(self, godown_code):
        """Get list of customers for a specific godown"""
        customers = self.customer_df[
            self.customer_df['godown_code'] == godown_code
        ]['cust_code'].unique().tolist()
        
        # Sort customers properly with '0' at the top if present
        if '0' in customers:
            customers.remove('0')
            sorted_customers = sorted(customers, key=lambda x: int(x) if x.isdigit() else float('inf'))
            sorted_customers.insert(0, '0')  # Add anonymous customer at beginning
            return sorted_customers
        else:
            return sorted(customers, key=lambda x: int(x) if x.isdigit() else float('inf'))

    def get_godowns(self):
        """Get list of all godowns with register code 7"""
        try:
            if self.customer_df is None:
                print("No customer data loaded")
                return []
            
            print("\nAnalyzing godown data:")
            print(f"Total records in customer_df: {len(self.customer_df)}")
            print(f"Unique godowns in customer_df: {len(self.customer_df['godown_code'].unique())}")
            print(f"All godown codes: {sorted(self.customer_df['godown_code'].unique())}")
            
            # Get godowns that have register code 7 transactions
            valid_godowns = self.customer_df[
                self.customer_df['register_code'] == '7'
            ]['godown_code'].unique()
            
            # Sort godowns for consistent ordering
            valid_godowns = sorted(valid_godowns)
            
            print(f"\nValid godowns analysis:")
            print(f"Number of valid godowns: {len(valid_godowns)}")
            print(f"Valid godown codes: {valid_godowns}")
            
            return valid_godowns
            
        except Exception as e:
            print(f"Error getting godowns: {str(e)}")
            traceback.print_exc()
            return [] 