var express = require('express');
var router = express.Router();
var { controllers } = require('./controllers');

/* GET */
router.get('/', controllers.main);
router.get('/current', controllers.main);
router.get('/saved', controllers.saved);
router.get('/services', controllers.services);
router.get('/about', controllers.about);


/* POST */
router.post('/api/generate', controllers.generate);

module.exports = router;
