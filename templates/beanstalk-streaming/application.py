# Author:  Meryll Dindin
# Date:    05 April 2020
# Project: Challenger

from imports import *

# Secure application
app = Flask(__name__)
app.secret_key = os.environ['FLASK_SECRET_KEY']
# Configure the websocket
wss = Sockets(app)

# Route to measure health response
@app.route('/health', methods=['GET'])
def health():

    arg = {'status': 200, 'mimetype': 'application/json'}
    return Response(response=json.dumps({'status': 'online'}), **arg)

@wss.route('/')
def stream(websocket):

    fle, cfg, t_0 = None, dict(), time.time()

    while not websocket.closed:

        # Catch the message
        msg = websocket.receive()

        if msg is None: continue

        else:
            msg = json.loads(msg)
            if msg['event'] == 'media':
                # Decode payload and write to opened file
                cks = base64.b64decode(msg['media']['payload'])
                if not fle is None: fle.write(bytes(cks))

                # TODO: 
                # Send the chunk of data to a ML-based thread for processing

            elif msg['event'] == 'start':
                _id = uuid.uuid4().hex[:32].lower()
                nme = '/'.join(['storage', '{}.{}'.format(_id, msg.get('format', 'ulaw'))])
                cfg.update({'fileid': _id, 'filepath': nme})
                fle = open(nme, 'wb+')
            elif msg['event'] == 'stop':
                break

    if not fle is None: fle.close()
    cfg.update({'completion': int(time.time() - t_0)})
    
    # TODO:
    # Add list of actions to execute on the serialized file
