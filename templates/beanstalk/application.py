# Author:  DINDIN Meryll
# Date:    23 Octobre 2019
# Project: beanstalk

from imports import *

# Secure application
application = Flask(__name__)
application.secret_key = os.environ['SECRET_KEY']

# Route to measure health response
@application.route('/health', methods=['GET'])
def health():

    return jsonify(status='online')

if __name__ == '__main__':

   arg = {'debug': True, 'threaded': True}
   application.run(host='127.0.0.1', port=8080, **arg)
