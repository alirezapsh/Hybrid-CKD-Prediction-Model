"""
Script to identify which rows in the ARFF file are skipped during parsing.
This finds the 3 rows that cause 400 -> 397 reduction.
"""

def find_skipped_rows():
    """Find which rows have incorrect number of values."""
    
    file_path = 'data/chronic_kidney_disease_full.arff'
    
    print("=" * 80)
    print("FINDING SKIPPED ROWS IN ARFF FILE")
    print("=" * 80)
    
    # Read file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Parse ARFF file manually (same logic as preprocessing.py)
    attributes = {}
    attribute_names = []
    data_started = False
    skipped_rows = []
    valid_rows = []
    
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith('%'):
            continue
        
        # Check for @attribute
        if stripped.lower().startswith('@attribute'):
            stripped_clean = stripped[len('@attribute'):].strip()
            
            if stripped_clean.startswith('"'):
                end_quote = stripped_clean.find('"', 1)
                if end_quote != -1:
                    attr_name = stripped_clean[1:end_quote].strip()
                else:
                    parts = stripped.split(None, 2)
                    if len(parts) >= 3:
                        attr_name = parts[1].strip()
                    else:
                        continue
            else:
                parts = stripped.split(None, 2)
                if len(parts) >= 3:
                    attr_name = parts[1].strip()
                else:
                    continue
            
            attr_name = attr_name.strip('"\'')
            attribute_names.append(attr_name)
            continue
        
        # Check for @data
        if stripped.lower().startswith('@data'):
            data_started = True
            continue
        
        # Parse data rows
        if data_started and stripped:
            # Split by comma and strip each value
            values = []
            for val in stripped.split(','):
                val_clean = val.strip()
                if not val_clean or val_clean == '?':
                    values.append(None)
                else:
                    values.append(val_clean)
            
            # Check if row has correct number of values
            if len(values) != len(attribute_names):
                skipped_rows.append({
                    'line_number': line_num,
                    'line_content': stripped[:100] + ('...' if len(stripped) > 100 else ''),
                    'expected_values': len(attribute_names),
                    'actual_values': len(values),
                    'values': values
                })
            else:
                valid_rows.append({
                    'line_number': line_num,
                    'values': values
                })
    
    print(f"\n1. ARFF File Analysis:")
    print(f"   Total attributes: {len(attribute_names)}")
    print(f"   Expected values per row: {len(attribute_names)}")
    print(f"   Valid data rows: {len(valid_rows)}")
    print(f"   Skipped rows: {len(skipped_rows)}")
    
    if skipped_rows:
        print(f"\n2. SKIPPED ROWS (These are the {len(skipped_rows)} removed samples):")
        print("=" * 80)
        
        for i, row_info in enumerate(skipped_rows, 1):
            print(f"\n   Row #{i}:")
            print(f"   Line number in ARFF file: {row_info['line_number']}")
            print(f"   Expected values: {row_info['expected_values']}")
            print(f"   Actual values: {row_info['actual_values']}")
            print(f"   Difference: {row_info['actual_values'] - row_info['expected_values']}")
            print(f"   Line content (first 100 chars): {row_info['line_content']}")
            
            # Try to identify which column is missing/extra
            if row_info['actual_values'] < row_info['expected_values']:
                print(f"   Issue: Row has FEWER values than expected (missing {row_info['expected_values'] - row_info['actual_values']} values)")
            elif row_info['actual_values'] > row_info['expected_values']:
                print(f"   Issue: Row has MORE values than expected (extra {row_info['actual_values'] - row_info['expected_values']} values)")
            
            # Show the values
            print(f"   Values in row: {row_info['values'][:10]}{'...' if len(row_info['values']) > 10 else ''}")
            if len(row_info['values']) > 0:
                # Try to identify the class value if it exists
                if len(row_info['values']) > len(attribute_names) - 1:
                    class_idx = len(attribute_names) - 1
                    if class_idx < len(row_info['values']):
                        print(f"   Class value (if present): '{row_info['values'][class_idx]}'")
        
        print("\n" + "=" * 80)
        print(f"\nSUMMARY:")
        print(f"   Total data rows in file: {len(valid_rows) + len(skipped_rows)}")
        print(f"   Valid rows (parsed): {len(valid_rows)}")
        print(f"   Skipped rows (removed): {len(skipped_rows)}")
        print(f"   Final count: {len(valid_rows)} (matches loaded samples)")
    else:
        print("\n   ✓ No rows skipped - all rows have correct number of values!")
    
    return skipped_rows, valid_rows

if __name__ == "__main__":
    skipped, valid = find_skipped_rows()

