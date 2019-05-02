var path = require('path');

// Init epxress lib.
var express = require('express');
var app = express();

// TODO I'd like to create a controllers factory impl.
var router = require('./router');

// Setting up template engine.
var mustacheExpress = require('mustache-express');
app.engine('html', mustacheExpress());
app.set('view engine', 'html');
app.set('views', path.join(__dirname, 'views'));

// Setting up the static folder, under media path.
app.use('/media', express.static('static'));

app.use('/', router)

app.listen(3000, function() {
  console.log("Server running at http://127.0.0.1:3000/");
});
