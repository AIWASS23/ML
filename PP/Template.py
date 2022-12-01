"""método para obter o texto do arquivo"""
def get_text():
     
    return "texto simples"
 
"""método para obter a versão xml do arquivo"""
def get_xml():
     
    return "xml"
 
"""método para obter a versão pdf do arquivo"""
def get_pdf():
     
    return "pdf"
 
"""método para obter a versão csv do arquivo"""
def get_csv():
     
    return "csv"
 
"""método usado para converter os dados em formato de texto"""
def convert_to_text(data):
     
    print("[CONVERTER]")
    return "{} como texto".format(data)
 
"""método usado para salvar os dados"""
def saver():
     
    print("[SALVE]")
 
"""função auxiliar nomeada como template_function"""
def template_function(getter, converter = False, to_save = False):
 
    """dados de entrada do getter"""
    data = getter()
    print("Conseguiu `{}`".format(data))
 
    if len(data) <= 3 and converter:
        data = converter(data)
    else:
        print("Pular conversão")
     
    """salva os dados apenas se o usuário quiser salvá-los"""
    if to_save:
        saver()
 
    print("`{}` foi processado".format(data))
 
 
"""main"""
if __name__ == "__main__":
 
    template_function(get_text, to_save = True)
 
    template_function(get_pdf, converter = convert_to_text)
 
    template_function(get_csv, to_save = True)
 
    template_function(get_xml, to_save = True)