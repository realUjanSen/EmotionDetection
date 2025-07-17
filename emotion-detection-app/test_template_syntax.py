#!/usr/bin/env python3
"""
Simple test to verify template syntax is correct
"""

def test_template_syntax():
    try:
        from jinja2 import Environment, FileSystemLoader
        
        # Create Jinja2 environment
        env = Environment(loader=FileSystemLoader('templates'))
        
        # Try to load and parse the template
        template = env.get_template('trends.html')
        print("✅ trends.html template loads successfully!")
        
        # Test a simple render (this will fail due to missing variables, but syntax should be ok)
        try:
            rendered = template.render(emotion_data={'labels': [], 'values': []})
            print("✅ trends.html template renders successfully!")
        except Exception as e:
            if "undefined" in str(e).lower():
                print("✅ trends.html template syntax is valid (missing variables is expected)")
            else:
                print(f"⚠️  Template render issue: {e}")
        
        return True
    except Exception as e:
        print(f"❌ Template syntax error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing trends.html template syntax...")
    success = test_template_syntax()
    if success:
        print("🎉 Template is ready!")
    else:
        print("❌ Template needs fixing.")
