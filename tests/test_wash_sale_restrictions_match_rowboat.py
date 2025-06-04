# import unittest
# from datetime import date, timedelta
# import pandas as pd
# import json
# from pathlib import Path
# from src.service.constraints.restriction.wash_sale_restrictions import WashSaleRestrictions
# from snowball.oracle_gateway.service.oracle_gateway import OracleGateway
# from src.service.oracle import Oracle


# class TestWashSaleRestrictionsMatchRowboat(unittest.TestCase):
#     """Test that our wash sale restrictions match OracleGateway's implementation."""
    
#     def setUp(self):
#         self.test_data_dir = Path(__file__).parent / 'rowboat_input_output'
#         self.gateway = OracleGateway()
        
#     def test_wash_sale_restrictions_match_rowboat(self):
#         """
#         For each .json file in the oracle_test_dumps folder:
#         1. Parse using OracleGateway
#         2. Create tax lots and everything for each strategy
#         3. Initialize wash sale restrictions
#         4. Compare our wash sale restrictions with OracleGateway's
#         """
#         # Process each JSON file in the test dumps directory in sorted order for consistency
#         for json_file in sorted(self.test_data_dir.glob('*.json')):
#             with open(json_file, 'r') as f:
#                 rowboat_data = json.load(f)
#                 layout = self.gateway.clear_nested_values(rowboat_data)
#                 input_data = rowboat_data.get('input', {})
#                 output_data = rowboat_data.get('output', {})
                
#             # Initialize Oracle through gateway
#             oracle = self.gateway.initialize_with_rowboat(input_data)
            
#             # Get the set of tickers we cannot buy according to Rowboat
#             rowboat_restricted_buys, rowboat_restricted_sells = self.gateway._extract_restrictions_from_rowboat_output(output_data)
            
#             # Get our wash sale restrictions
#             double_restricted_buys = oracle.wash_sale_restrictions.get_all_restricted_buys()
#             double_restricted_sells = oracle.wash_sale_restrictions.get_all_restricted_sells()

#             # Check if double's restricted buys are a subset of rowboat's
#             self.assertTrue(
#                 double_restricted_buys.issubset(rowboat_restricted_buys),
#                 f"Double's restricted buys should be a subset of Rowboat's in {json_file.name}.\n"
#                 f"Double has extra tickers: {double_restricted_buys - rowboat_restricted_buys}\n"
#                 f"Double: {double_restricted_buys}\n"
#                 f"Rowboat: {rowboat_restricted_buys}"
#             )
            
#             # We don't need to check for extra tickers in Oracle anymore since we expect Rowboat to have more
#             # But we'll log the difference for visibility
#             extra_in_rowboat = rowboat_restricted_buys - double_restricted_buys
#             if extra_in_rowboat:
#                 print(f"Note: Rowboat has additional restricted tickers (expected): {extra_in_rowboat} in {json_file.name}")
            
#             # Handle empty DataFrames gracefully for sells comparison
#             if (isinstance(rowboat_restricted_sells, pd.DataFrame) and rowboat_restricted_sells.empty) and \
#                (isinstance(double_restricted_sells, pd.DataFrame) and double_restricted_sells.empty):
#                 # Both are empty - this is a valid match
#                 rowboat_identifiers = set()
#                 double_identifiers = set()
#             else:
#                 # Sort both DataFrames by identifier for consistent comparison
#                 # If one is empty, create an empty DataFrame with matching columns
#                 if isinstance(rowboat_restricted_sells, pd.DataFrame) and rowboat_restricted_sells.empty:
#                     rowboat_restricted_sells = pd.DataFrame(columns=double_restricted_sells.columns)
#                 if isinstance(double_restricted_sells, pd.DataFrame) and double_restricted_sells.empty:
#                     double_restricted_sells = pd.DataFrame(columns=rowboat_restricted_sells.columns)
                
#                 rowboat_sells = rowboat_restricted_sells.sort_values('identifier').reset_index(drop=True)
#                 double_sells = double_restricted_sells.sort_values('identifier').reset_index(drop=True)
                
#                 rowboat_identifiers = set(rowboat_sells['identifier'])
#                 double_identifiers = set(double_sells['identifier'])

#             extra_identifiers = double_identifiers - rowboat_identifiers
#             if extra_identifiers:
#                 self.fail(f"Oracle has extra identifiers in restricted sells: {extra_identifiers}")
            
#             # For each identifier, check if all tax lots match
#             for identifier in rowboat_identifiers:
#                 rowboat_lots = rowboat_sells[rowboat_sells['identifier'] == identifier][["identifier", "quantity", "date_acquired"]].sort_values(
#                     by=['identifier', 'date_acquired', 'quantity']
#                 ).reset_index(drop=True)
#                 double_lots = double_sells[double_sells['identifier'] == identifier][["identifier", "quantity", "date_acquired"]].sort_values(
#                     by=['identifier', 'date_acquired', 'quantity']
#                 ).reset_index(drop=True)

#                 for double_lot in double_lots.to_dict('records'):
#                     exists = rowboat_lots[
#                         (rowboat_lots['identifier'] == double_lot['identifier']) &
#                         (pd.to_datetime(rowboat_lots['date_acquired']) == pd.to_datetime(double_lot['date_acquired'])) &
#                         (rowboat_lots['quantity'] == double_lot['quantity'])
#                     ]
                    
#                     if exists.empty:
#                         self.fail(f"Rowboat is missing tax lot: {double_lot} for {identifier} in {json_file.name}")
            
#                 # Rowboat adds things to restirctions based on it's current trades. 
#                 # So ours is a subset of theirs.
#                 # for rowboat_lot in rowboat_lots.to_dict('records'):
#                 #     exists = double_lots[
#                 #         (double_lots['identifier'] == rowboat_lot['identifier']) &
#                 #         (pd.to_datetime(double_lots['date_acquired']) == pd.to_datetime(rowboat_lot['date_acquired'])) &
#                 #         (double_lots['quantity'] == rowboat_lot['quantity'])
#                 #     ]
                    
#                 #     if exists.empty:
#                 #         self.fail(f"Oracle is missing tax lot: {rowboat_lot} for {identifier} in {json_file.name}")
            
#             # Compare restricted sells DataFrames
#             # First check if the DataFrames have the same number of rows
#             # self.assertEqual(
#             #     len(rowboat_restricted_sells),
#             #     len(double_restricted_sells),
#             #     f"Different number of restricted sells. Rowboat: {len(rowboat_restricted_sells)}, "
#             #     f"Oracle: {len(double_restricted_sells)} in {json_file.name}"
#             # )