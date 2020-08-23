function yesnoCheck() {
    var element = document.querySelectorAll('[id=matrixPertinence]');
    var elementInput = document.querySelectorAll('[id=matrixPertinenceValue]');
    if (document.getElementById('yesCheck').checked) {

        for(var i = 0; i < element.length; i++){element[i].style.display = 'block'; }
        for(var i = 0; i < elementInput.length; i++){elementInput[i].required = true; }
    }
    else {
        for(var i = 0; i < element.length; i++){element[i].style.display = 'none';}
        for(var i = 0; i < elementInput.length; i++){elementInput[i].required = false;}
    }
}

function changeAssignment(){
    var selectBox = document.getElementById("selectBox");
    var selectedValue = selectBox.options[selectBox.selectedIndex].value;
    if(selectedValue == '1')
    {
        document.getElementById("weight").style.display = 'block';
    }else{
        document.getElementById("weight").style.display = 'none';
    }
}