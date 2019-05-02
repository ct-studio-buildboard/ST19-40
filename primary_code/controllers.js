module.exports.controllers = {
  // GET Controllers
  main: function(req, res) {

    res.render('main.html', {conversionOptions: [{name: 'test', value: 'test'}, {name: 'test1', value: 'test1'}]});
  },
  saved: function(req, res) {

    res.send('saved');
  },
  services: function(req, res) {
    res.send('services');
  },
  about: function(req, res) {
    res.send('about');
  },
  // POST controllers
  generate: function(req, res) {
    res.send('Generates');
  }
}
