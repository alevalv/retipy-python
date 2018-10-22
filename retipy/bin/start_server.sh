#!/bin/sh

export FLASK_APP=retipy-server
export FLASK_DEBUG=true
exec flask run
