general_var=null;
general=null;
console.log("WRF THE HELL");
$( document ).ready(function() {
  $('#inputGroupFile').on('change',function(){
      //get the file name
      var fileName = $(this).val();
      var fileNameParts =  fileName.split('\\');
      //replace the "Choose a file" label
      $(this).next('.custom-file-label').html(fileNameParts.pop());
      previewSourceImg(this);


  });
whut=null;
  function previewSourceImg(input) {
    if (input.files && input.files[0]) {
      var reader = new FileReader();
        reader.onload = function (e) {
          $('#source-img').attr('src', e.target.result);
          general=e.target.result;
         // whut=e.target
        }
        reader.readAsDataURL(input.files[0]);

    }
  }

  $('.btn-primary').click(function(){

      //general=encodeURL(general)
      generate_image();

   //$('#final_image').attr('src',"static/gif_x.gif");
      setTimeout(updateImageOutputted, 800);
      // $('#final_image').src("static/x_final.jpg" )
      //print("WTF");
      //console.log(whut);
  });

 function encodeURL(str){
     return str ;
    //return str.replace(/\+/g, '-').replace(/\//g, '_').replace(/\=+$/, '');
}

function updateImageOutputted(){
     $('#final_image').attr('src',"static/x_final.jpg?version="+Math.random().toString()  );
}


});




function generate_image(){


            $.ajax({
                url: "api/generate_image",
                type: "POST",
                data: 'jsonData=' + JSON.stringify({'user_id':general}),
                success: function (response) {
                  console.log('unsure messages Receives'+response);


                }.bind(this)

            });
        }

