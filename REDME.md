# Acme Food Bank Inventory Management System

## Project Summary

### One-sentence description of the project
A secure, real-time inventory management system designed to track and manage perishable and non-perishable donations for food banks.

### Additional information about the project
The Acme Food Bank Inventory Management System provides a comprehensive platform for managing incoming donations, monitoring stock levels, setting thresholds, and identifying soon-to-expire items. With features like two-factor authentication, role-based access, and dynamic dashboard visualizations, the system enhances operational efficiency, reduces waste, and supports data-driven decisions for food distribution. The system supports both administrative users and general volunteers through secure login and custom access levels.

## Installation

### Prerequisites
- Python 3.10+
- pip
- Git
- SQLite3 (default, included with Python)
- Recommended: virtualenv

### Add-ons
- **Django** – Main web framework
- **django-crispy-forms** – For clean form rendering
- **Bootstrap 5** – For UI styling
- **Chart.js or Matplotlib** – For dashboard visualizations
- **Selenium** – For automated browser-based testing
- **JWT (via rest_framework_simplejwt)** – For secure session/token management

### Installation Steps
```bash
# Clone the repository
git clone https://github.com/shamney/Shampurnad.github.io.git

# Navigate to the project directory
cd Shampurnad.github.io

# Switch to the desired branch
git checkout unit-testing

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py migrate

# Create a superuser (for admin login)
python manage.py createsuperuser

# Start the development server
python manage.py runserver

#for selenium testing
export TESTING=1
python manage.py test apps.user_management.tests_selenium

#for unit testing
python manage.py test apps.inventory
python manage.py test apps.user_management
