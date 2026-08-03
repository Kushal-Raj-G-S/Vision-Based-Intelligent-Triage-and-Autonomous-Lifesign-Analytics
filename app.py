import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Add current directory to python path for package imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from web import app

if __name__ == '__main__':
    print("=" * 60)
    print("VITAL - Opti-Screen SaaS API Server")
    print("=" * 60)
    
    print("  Frontend (Next.js): http://localhost:3000")
    print("  API Backend:        http://localhost:5002")
    print("  Open http://localhost:3000 in your browser.")
    print("=" * 60)
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=5002, threaded=True, use_reloader=False)
