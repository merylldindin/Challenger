# Author:  Meryll Dindin
# Date:    05 April 2020
# Project: Challenger

from application import app, wss

if __name__ == '__main__':

    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    # Development server to run locally on port 5000
    app = pywsgi.WSGIServer(('', 5000), wss, handler_class=WebSocketHandler)
    app.serve_forever()